import os

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from functools import partial
import numpy as np
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx 
import jax
import time

class cem_planner():

	def __init__(self, num_dof=None, num_batch=None, num_steps=None, timestep=None, maxiter_cem=None, num_elite=None, w_pos=None, w_rot=None, w_col=None, 
			     maxiter_projection=None, max_pos = None ,max_vel = None, 
				 max_acc = None, max_jerk = None):
		super(cem_planner, self).__init__()
	 
		self.num_dof = num_dof
		self.num_batch = num_batch
		self.t = timestep
		self.num_steps = num_steps
		self.num_elite = num_elite
		self.cost_weights = {
			'w_pos': w_pos,
			'w_rot': w_rot,
			'w_col': w_col,
		}

		self.t_fin = self.num_steps*self.t
		
		tot_time = np.linspace(0, self.t_fin, self.num_steps)
		self.tot_time = tot_time
		tot_time_copy = tot_time.reshape(self.num_steps, 1)

        # Velocity mapping (identity for direct velocity control)
		self.P = jnp.identity(self.num_steps)
		
		# Acceleration mapping
		self.Pdot = jnp.diff(self.P, axis=0)/self.t
        
		# Jerk mapping
		self.Pddot = jnp.diff(self.Pdot, axis=0)/self.t

        # Position mapping (integration of velocity)
		self.Pint = jnp.cumsum(self.P, axis=0)*self.t
		
		self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)
		self.Pint_jax = jnp.asarray(self.Pint)

		self.nvar_single = jnp.shape(self.P_jax)[1]
		self.nvar = self.nvar_single*self.num_dof 
  
		self.rho_ineq = 1.0
		self.rho_projection = 1.0

		self.A_projection_single_dof = jnp.identity(self.nvar_single)

		A_v_ineq_single_dof, A_v_single_dof = self.get_A_v_single_dof()
		self.A_v_ineq_single_dof = jnp.asarray(A_v_ineq_single_dof) 
		self.A_v_single_dof = jnp.asarray(A_v_single_dof)

		A_a_ineq_single_dof, A_a_single_dof = self.get_A_a_single_dof()
		self.A_a_ineq_single_dof = jnp.asarray(A_a_ineq_single_dof) 
		self.A_a_single_dof = jnp.asarray(A_a_single_dof)

		A_j_ineq_single_dof, A_j_single_dof = self.get_A_j_single_dof()
		self.A_j_ineq_single_dof = jnp.asarray(A_j_ineq_single_dof)
		self.A_j_single_dof = jnp.asarray(A_j_single_dof)
  
		A_p_ineq_single_dof, A_p_single_dof = self.get_A_p_single_dof()
		self.A_p_ineq_single_dof = jnp.asarray(A_p_ineq_single_dof) 
		self.A_p_single_dof = jnp.asarray(A_p_single_dof)

		# Combined control matrix
		self.A_control_single_dof = jnp.vstack((
			self.A_v_ineq_single_dof,
			self.A_a_ineq_single_dof,
			self.A_j_ineq_single_dof,
			self.A_p_ineq_single_dof
		))

		A_eq_single_dof = self.get_A_eq_single_dof()
		self.A_eq_single_dof = jnp.asarray(A_eq_single_dof)

		A_theta, A_thetadot, A_thetaddot, A_thetadddot = self.get_A_traj()

		self.A_theta = np.asarray(A_theta)
		self.A_thetadot = np.asarray(A_thetadot)
		self.A_thetaddot = np.asarray(A_thetaddot)
		self.A_thetadddot = np.asarray(A_thetadddot)
		
		self.compute_boundary_vec_batch_single_dof = (jax.vmap(self.compute_boundary_vec_single_dof, in_axes = (0)  ))

		self.key= jax.random.PRNGKey(42)
		self.maxiter_projection = maxiter_projection
		self.maxiter_cem = maxiter_cem

		# FIXED: Increased velocity limits to allow higher control values
		self.v_max = max_vel  # This should be higher, e.g., 10.0 or 20.0
		self.a_max = max_acc
		self.j_max = max_jerk
		self.p_max = max_pos

		# Calculate number constraints
		self.num_vel = self.num_steps
		self.num_acc = self.num_steps - 1
		self.num_jerk = self.num_steps - 2
		self.num_pos = self.num_steps

		self.num_vel_constraints = 2 * self.num_vel * num_dof
		self.num_acc_constraints = 2 * self.num_acc * num_dof
		self.num_jerk_constraints = 2 * self.num_jerk * num_dof
		self.num_pos_constraints = 2 * self.num_pos * num_dof
		self.num_total_constraints = (self.num_vel_constraints + self.num_acc_constraints + 
									 	self.num_jerk_constraints + self.num_pos_constraints)
		
		self.num_total_constraints_per_dof = 2*(self.num_vel + self.num_acc + self.num_jerk + self.num_pos)

		self.ellite_num = int(self.num_elite*self.num_batch)

		self.alpha_mean = 0.8
		self.alpha_cov = 0.8

		self.lamda = 1
		self.g = 10
		self.vec_product = jax.jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))
		
		self.model_path = f"{os.path.dirname(__file__)}/default4.xml" 
		self.model = mujoco.MjModel.from_xml_path(self.model_path)
		self.data = mujoco.MjData(self.model)
		self.model.opt.timestep = self.t

		self.mjx_model = mjx.put_model(self.model)
		self.mjx_data = mjx.put_data(self.model, self.data)
		self.mjx_data = jax.jit(mjx.forward)(self.mjx_model, self.mjx_data)
		self.jit_step = jax.jit(mjx.step)

		self.geom_ids = []
		
		for i in range(self.model.ngeom):
			name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
			if name is not None and name.startswith('robot'):  
				self.geom_ids.append(i)

		self.geom_ids_all = np.array(self.geom_ids)
		self.mask = jnp.any(jnp.isin(self.mjx_data.contact.geom, self.geom_ids_all), axis=1)

		self.base_id = self.model.body(name="robot0:base_link").id

		self.compute_rollout_batch = jax.vmap(self.compute_rollout_single, in_axes = (0))
		self.compute_cost_batch = jax.vmap(self.compute_cost_single, in_axes = (0, 0, None))
		self.compute_projection_batched_over_dof = jax.vmap(self.compute_projection_single_dof, in_axes=(0, 0, 0, 0, 0))

		self.print_info()

	def print_info(self):
		print(
			f'\n Default backend: {jax.default_backend()}'
			f'\n Model path: {self.model_path}',
			f'\n Timestep: {self.t}',
			f'\n CEM Iter: {self.maxiter_cem}',
			f'\n Projection Iter: {self.maxiter_projection}',
			f'\n Number of batches: {self.num_batch}',
			f'\n Number of steps per trajectory: {self.num_steps}',
			f'\n Time per trajectory: {self.t_fin}',
			f'\n Number of variables: {self.nvar}',
			f'\n Number of Total constraints: {self.num_total_constraints}',
			f'\n Max velocity: {self.v_max}',
			f'\n Max acceleration: {self.a_max}',
			f'\n Max position: {self.p_max}'
		)

	def get_A_traj(self):
		# For direct velocity control
		A_theta = np.kron(np.identity(self.num_dof), self.Pint )
		A_thetadot = np.kron(np.identity(self.num_dof), self.P )
		A_thetaddot = np.kron(np.identity(self.num_dof), self.Pdot )
		A_thetadddot = np.kron(np.identity(self.num_dof), self.Pddot )
		return A_theta, A_thetadot, A_thetaddot, A_thetadddot	

	def get_A_p_single_dof(self):
		A_p = np.vstack(( self.Pint, -self.Pint))
		A_p_ineq = np.kron(np.identity(1), A_p )
		return A_p_ineq, A_p
	
	def get_A_v_single_dof(self):
		A_v = np.vstack(( self.P, -self.P     ))
		A_v_ineq = np.kron(np.identity(1), A_v )
		return A_v_ineq, A_v

	def get_A_a_single_dof(self):
		A_a = np.vstack(( self.Pdot, -self.Pdot  ))
		A_a_ineq = np.kron(np.identity(1), A_a )
		return A_a_ineq, A_a
	
	def get_A_j_single_dof(self):
		A_j = np.vstack(( self.Pddot, -self.Pddot  ))
		A_j_ineq = np.kron(np.identity(1), A_j )
		return A_j_ineq, A_j
	
	def get_A_eq_single_dof(self):
		return np.kron(np.identity(1), self.P[0])

	@partial(jax.jit, static_argnums=(0,))
	def compute_boundary_vec_single_dof(self, state_term):
		num_eq_constraint_per_dof = int(jnp.shape(state_term)[0])
		b_eq_term = state_term.reshape( num_eq_constraint_per_dof).T
		b_eq_term = b_eq_term.reshape(num_eq_constraint_per_dof)
		return b_eq_term

	@partial(jax.jit, static_argnums=(0,))
	def compute_feasible_control_single_dof(self, lamda_init_single_dof, s_init_single_dof, 
										 b_eq_term_single_dof, xi_samples_single_dof, 
										 init_pos_single_dof):
		b_vel = jnp.hstack((
			self.v_max * jnp.ones((self.num_batch, self.num_vel_constraints // (2*self.num_dof))),
			self.v_max * jnp.ones((self.num_batch, self.num_vel_constraints // (2*self.num_dof)))
		))

		b_acc = jnp.hstack((
			self.a_max * jnp.ones((self.num_batch, self.num_acc_constraints // (2*self.num_dof))),
			self.a_max * jnp.ones((self.num_batch, self.num_acc_constraints // (2*self.num_dof)))
		))

		b_jerk = jnp.hstack((
			self.j_max * jnp.ones((self.num_batch, self.num_jerk_constraints // (2*self.num_dof))),
			self.j_max * jnp.ones((self.num_batch, self.num_jerk_constraints // (2*self.num_dof)))
		))

		init_pos_single_dof_batch = jnp.tile(init_pos_single_dof, (self.num_batch, 1))
		
		b_pos_upper = (self.p_max - init_pos_single_dof_batch)
		b_pos_lower = (self.p_max + init_pos_single_dof_batch)
		
		b_pos_upper_expanded = jnp.tile(b_pos_upper[:, :, None], (1, 1, self.num_pos_constraints // (self.num_dof * 2)))
		b_pos_lower_expanded = jnp.tile(b_pos_lower[:, :, None], (1, 1, self.num_pos_constraints // (self.num_dof * 2)))
		
		b_pos_stacked = jnp.concatenate([b_pos_upper_expanded, b_pos_lower_expanded], axis=2)
		b_pos = b_pos_stacked.reshape((self.num_batch, -1))
        
		b_control_single_dof = jnp.hstack((b_vel, b_acc, b_jerk, b_pos))

		# Augmented bounds with slack variables
		b_control_aug_single_dof = b_control_single_dof - s_init_single_dof

		# Cost matrix
		cost = (
			jnp.dot(self.A_projection_single_dof.T, self.A_projection_single_dof) +
			self.rho_ineq * jnp.dot(self.A_control_single_dof.T, self.A_control_single_dof)
		)

		# KKT system matrix
		cost_mat = jnp.vstack((
			jnp.hstack((cost, self.A_eq_single_dof.T)),
			jnp.hstack((self.A_eq_single_dof, jnp.zeros((self.A_eq_single_dof.shape[0], self.A_eq_single_dof.shape[0]))))
		))

		# Linear cost term
		lincost = (
			-lamda_init_single_dof -
			jnp.dot(self.A_projection_single_dof.T, xi_samples_single_dof.T).T -
			self.rho_ineq * jnp.dot(self.A_control_single_dof.T, b_control_aug_single_dof.T).T
		)

		# Solve KKT system
		sol = jnp.linalg.solve(cost_mat, jnp.hstack((-lincost, b_eq_term_single_dof)).T).T

		# Extract primal solution
		xi_projected = sol[:, :self.nvar_single]

		# Update slack variables
		s = jnp.maximum(
			jnp.zeros((self.num_batch, self.num_total_constraints_per_dof)),
			-jnp.dot(self.A_control_single_dof, xi_projected.T).T + b_control_single_dof
		)

		# Compute residual
		res_vec = jnp.dot(self.A_control_single_dof, xi_projected.T).T - b_control_single_dof + s
		res_norm = jnp.linalg.norm(res_vec, axis=1)

		# Update Lagrange multipliers
		lamda = lamda_init_single_dof - self.rho_ineq * jnp.dot(self.A_control_single_dof.T, res_vec.T).T

		return xi_projected, s, res_norm, lamda

	@partial(jax.jit, static_argnums=(0,))
	def compute_projection_single_dof(self, 
								       xi_samples_single_dof, 
								       state_term_single_dof, 
									   lamda_init_single_dof, 
									   s_init_single_dof, 
									   init_pos_single_dof):
		b_eq_term = self.compute_boundary_vec_batch_single_dof(state_term_single_dof)
		xi_projected_init_single_dof = xi_samples_single_dof

		def lax_custom_projection(carry, idx):
			_, lamda, s = carry
			lamda_prev, s_prev = lamda, s
			
			primal_sol, s, res_projection, lamda = self.compute_feasible_control_single_dof(lamda, 
																		s, b_eq_term, xi_samples_single_dof, 
																		init_pos_single_dof)
			
			primal_residual = res_projection
			fixed_point_residual = (
				jnp.linalg.norm(lamda_prev - lamda, axis=1) +
				jnp.linalg.norm(s_prev - s, axis=1)
			)
			return (primal_sol, lamda, s), (primal_residual, fixed_point_residual)

		carry_init = (xi_projected_init_single_dof, lamda_init_single_dof, s_init_single_dof)

		carry_final, res_tot = jax.lax.scan(
			lax_custom_projection,
			carry_init,
			jnp.arange(self.maxiter_projection)
		)

		primal_sol, lamda, s = carry_final
		primal_residuals, fixed_point_residuals = res_tot

		primal_residuals = jnp.stack(primal_residuals)
		fixed_point_residuals = jnp.stack(fixed_point_residuals)

		return primal_sol, primal_residuals, fixed_point_residuals
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_rollout_single(self, ctrl):
		mjx_data = self.mjx_data
		ctrl_single = ctrl.reshape(self.num_dof, self.num_steps)
		_, out = jax.lax.scan(self.mjx_step, mjx_data, ctrl_single.T, length=self.num_steps)
		base_pos, base_angvel = out
		return base_pos, base_angvel
	
	@partial(jax.jit, static_argnums=(0,))
	def mjx_step(self, mjx_data, ctrl_single):
		# FIXED: Increased clipping range to allow higher control values
		ctrl_single = jnp.clip(ctrl_single, -200.0, 200.0)  # Increased from -50, 50
		mjx_data = mjx_data.replace(
            ctrl=mjx_data.ctrl.at[:self.num_dof].set(ctrl_single)
        )
		mjx_data = self.jit_step(self.mjx_model, mjx_data)
		base_pos = mjx_data.xpos[self.base_id]
		base_angvel = mjx_data.cvel[self.base_id, 3:6]
		return mjx_data, (base_pos, base_angvel)
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_cost_single(self, base_pos, base_angvel, target_pos):
		target_x = target_pos[-1, 0]
		base_x = base_pos[-1, 0]
		cost_pos = jnp.linalg.norm(base_x - target_x)
		cost_angvel = jnp.sum(jnp.abs(base_angvel))
		# FIXED: Increased position cost weight to encourage forward movement
		cost = 10.0 * cost_pos + 0.1 * cost_angvel  # Increased from 0.0 to 1.0
		return cost
	
	@partial(jax.jit, static_argnums=(0, ))
	def compute_ellite_samples(self, cost_batch, xi_filtered):
		idx_ellite = jnp.argsort(cost_batch)
		cost_ellite = cost_batch[idx_ellite[0:self.ellite_num]]
		xi_ellite = xi_filtered[idx_ellite[0:self.ellite_num]]
		return xi_ellite, idx_ellite, cost_ellite
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_xi_samples(self, key, xi_mean, xi_cov ):
		key, subkey = jax.random.split(key)
		xi_samples = jax.random.multivariate_normal(key, xi_mean, xi_cov+0.1*jnp.identity(self.nvar), (self.num_batch, ))
		return xi_samples, key
	
	@partial(jax.jit, static_argnums=(0,))
	def comp_prod(self, diffs, d ):
		term_1 = jnp.expand_dims(diffs, axis = 1)
		term_2 = jnp.expand_dims(diffs, axis = 0)
		prods = d * jnp.outer(term_1,term_2)
		return prods	
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_mean_cov(self, cost_ellite, mean_control_prev, cov_control_prev, xi_ellite):
		w = cost_ellite
		w_min = jnp.min(cost_ellite)
		w = jnp.exp(-(1/self.lamda) * (w - w_min ) )
		sum_w = jnp.sum(w, axis = 0)
		mean_control = (1-self.alpha_mean)*mean_control_prev + self.alpha_mean*(jnp.sum( (xi_ellite * w[:,jnp.newaxis]) , axis= 0)/ sum_w)
		diffs = (xi_ellite - mean_control)
		prod_result = self.vec_product(diffs, w)
		cov_control = (1-self.alpha_cov)*cov_control_prev + self.alpha_cov*(jnp.sum( prod_result , axis = 0)/jnp.sum(w, axis = 0)) + 0.0001*jnp.identity(self.nvar)
		return mean_control, cov_control
	
	@partial(jax.jit, static_argnums=(0,))
	def cem_iter(self, carry,  scan_over):
		init_pos, init_vel, target_pos, target_rot, xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples = carry

		xi_mean_prev = xi_mean 
		xi_cov_prev = xi_cov

		xi_samples_reshaped = xi_samples.reshape(self.num_batch, self.num_dof, self.num_steps)
		xi_samples_batched_over_dof = jnp.transpose(xi_samples_reshaped, (1, 0, 2))

		state_term_reshaped = state_term.reshape(self.num_batch, self.num_dof, 1)
		state_term_batched_over_dof = jnp.transpose(state_term_reshaped, (1, 0, 2))

		lamda_init_reshaped = lamda_init.reshape(self.num_batch, self.num_dof, self.num_steps)
		lamda_init_batched_over_dof = jnp.transpose(lamda_init_reshaped, (1, 0, 2))

		s_init_reshaped = s_init.reshape(self.num_batch, self.num_dof, self.num_total_constraints_per_dof )
		s_init_batched_over_dof = jnp.transpose(s_init_reshaped, (1, 0, 2))

		xi_filtered, primal_residuals, fixed_point_residuals = self.compute_projection_batched_over_dof(
			                                                     xi_samples_batched_over_dof, 
														         state_term_batched_over_dof, 
																 lamda_init_batched_over_dof, 
																 s_init_batched_over_dof, 
																 init_pos)
		
		xi_filtered = xi_filtered.transpose(1, 0, 2).reshape(self.num_batch, -1)
		
		primal_residuals = jnp.linalg.norm(primal_residuals, axis = 0)
		fixed_point_residuals = jnp.linalg.norm(fixed_point_residuals, axis = 0)
				
		avg_res_primal = jnp.sum(primal_residuals, axis = 0)/self.maxiter_projection
		avg_res_fixed_point = jnp.sum(fixed_point_residuals, axis = 0)/self.maxiter_projection
		
		ctrl = xi_filtered
		base_pos, base_angvel = self.compute_rollout_batch(ctrl)
		cost_batch = self.compute_cost_batch(base_pos, base_angvel, target_pos)

		xi_ellite, idx_ellite, cost_ellite = self.compute_ellite_samples(cost_batch, xi_samples)
		xi_mean, xi_cov = self.compute_mean_cov(cost_ellite, xi_mean_prev, xi_cov_prev, xi_ellite)

		xi_samples_new, key = self.compute_xi_samples(key, xi_mean, xi_cov)

		carry = (init_pos, init_vel, target_pos, target_rot, xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples_new)

		return carry, (cost_batch, ctrl, base_pos, avg_res_primal, avg_res_fixed_point, primal_residuals, fixed_point_residuals)

	@partial(jax.jit, static_argnums=(0,))
	def compute_cem(
		self, xi_mean, 
		xi_cov,
		init_pos, 
		init_vel, 
		init_acc,
		target_pos,
		target_rot,
		lamda_init,
		s_init,
		xi_samples
		):

		vel_init = jnp.tile(init_vel, (self.num_batch, 1))
		target_pos = jnp.tile(target_pos, (self.num_batch, 1))
		target_rot = jnp.tile(target_rot, (self.num_batch, 1))
		
		state_term = vel_init	
		key, subkey = jax.random.split(self.key)

		carry = (init_pos, init_vel, target_pos, target_rot, xi_mean, xi_cov, key, state_term, lamda_init, s_init, xi_samples)
		scan_over = jnp.array([0]*self.maxiter_cem)
		
		carry, out = jax.lax.scan(self.cem_iter, carry, scan_over, length=self.maxiter_cem)
		cost_batch, ctrl, base_pos, avg_res_primal, avg_res_fixed, primal_residuals, fixed_point_residuals = out

		idx_min = jnp.argmin(cost_batch[-1])
		cost = jnp.min(cost_batch, axis=1)
		best_ctrl = ctrl[-1][idx_min].reshape((self.num_dof, self.num_steps)).T
		best_traj = base_pos[-1][idx_min].reshape((self.num_steps, 3))

		xi_mean = carry[4]
		xi_cov = carry[5]

		return (
			cost,
			best_ctrl,
			best_traj,
			xi_mean,
			xi_cov,
			ctrl,
			base_pos,
			avg_res_primal,
			avg_res_fixed,
			primal_residuals,
			fixed_point_residuals,
			idx_min,
		)

def main():
	num_dof = 2
	num_batch = 2000

	start_time = time.time()
	opt_class = cem_planner(num_dof=num_dof, num_batch=num_batch, num_steps=50, maxiter_cem=30,
                           w_pos=1, w_rot=0.5, w_col=10, num_elite=0.2, timestep=0.05,
						   maxiter_projection=5, max_pos = np.pi, max_vel=100.0, max_acc=600.0, max_jerk=60.0)

	start_time_comp_cem = time.time()
	xi_mean = jnp.zeros(opt_class.nvar)
	xi_cov = 100.0*jnp.identity(opt_class.nvar)
	xi_samples, key = opt_class.compute_xi_samples(opt_class.key, xi_mean, xi_cov)
	init_pos = jnp.array([0.0, 0.0])
	init_vel = jnp.array([20.0, 0.0]) #linear velocity and angular velocity
	init_acc = jnp.array([0.0])
	target_pos = jnp.array([5.5, 0, 0.0])
	target_rot = jnp.array([0.0, 0.0, 0.0, 1.0])
	s_init = jnp.zeros((opt_class.num_batch, opt_class.num_total_constraints))
	lamda_init = jnp.zeros((opt_class.num_batch, opt_class.nvar))
	
	(cost,
    best_ctrl,
    best_traj,
    xi_mean,
    xi_cov,
    ctrl,
    base_pos,
    avg_res_primal,
    avg_res_fixed,
    primal_residuals,
    fixed_point_residuals,
    idx_min) = opt_class.compute_cem(
		xi_mean,
		xi_cov,
		init_pos,
		init_vel,
		init_acc,
		target_pos,
		target_rot,
		lamda_init,
		s_init,
		xi_samples
	)
	
	print(f"Total time: {round(time.time()-start_time, 2)}s")
	print(f"Compute CEM time: {round(time.time()-start_time_comp_cem, 2)}s")
	print(f"ctrl: {ctrl}")

	# 1. Define a filename for the output
	output_file = 'best_control_sequence.npy'

	# 2. Convert the JAX array 'best_ctrl' to a NumPy array
	#    Your 'best_ctrl' variable already has the shape (num_steps, num_dof)
	best_ctrl_numpy = np.asarray(best_ctrl)

	# 3. Save the NumPy array to the specified file
	np.save(output_file, best_ctrl_numpy)

	# 4. Print a confirmation message to the console
	print(f"\n--- Best control sequence saved to '{output_file}' ---")
	print(f"Shape of saved controls: {best_ctrl_numpy.shape}")
	print(f"Sample of saved controls (first 5 steps):\n{best_ctrl_numpy[:5,:]}")

	
	
if __name__ == "__main__":
	main()


  	