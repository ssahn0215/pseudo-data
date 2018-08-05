import tensorflow as tf

def mala(thetas, log_probs_fn, lr, nb_walk_per_step):
    avg_acceptance_rate_ = 0.0
    thetas_ = thetas
    log_probs_ = log_probs_fn(thetas_)
    grads_ = tf.gradients(log_probs_, thetas_)[0]
    for step in range(nb_walk_per_step):
        eps = tf.random_normal(thetas_.get_shape())
        proposed_thetas_ = tf.stop_gradient(thetas_+lr*grads_+tf.sqrt(2*lr)*eps)
        proposed_log_probs_ = log_probs_fn(proposed_thetas_)
        proposed_grads_ = tf.gradients(proposed_log_probs_, proposed_thetas_)[0]

        # add rejection step
        log_numer = proposed_log_probs_-0.25/lr*tf.reduce_sum(
            tf.square(thetas_-proposed_thetas_-lr*proposed_grads_), axis=1)
        log_denom = log_probs_-0.5*tf.reduce_sum(tf.square(eps), axis=1)
        acceptance_rate = tf.clip_by_value(tf.exp(log_numer-log_denom), 0.0, 1.0)

        # accept samples and update related quantities
        u = tf.random_uniform(acceptance_rate.get_shape())
        accept = tf.less_equal(u, acceptance_rate)
        thetas_ = tf.where(accept, proposed_thetas_, thetas_)
        log_probs_ = tf.where(accept, proposed_log_probs_, log_probs_)

        avg_acceptance_rate_ += tf.reduce_mean(acceptance_rate)/nb_walk_per_step
        if step < nb_walk_per_step-1:
            grads_ = tf.where(accept, proposed_grads_, grads_)

    return tf.assign(thetas, thetas_), avg_acceptance_rate_

def langevin_step(thetas, log_probs_fn, lr):
    grads = tf.gradients(log_probs_fn(thetas), thetas)
    eps = tf.random_normal(thetas.get_shape())
    return tf.assign_add(thetas, lr*grads+tf.sqrt(2*lr)*eps)

def gradient_descent_step(thetas, log_probs_fn, lr):
    grads = tf.gradients(log_probs_fn(thetas), thetas)
    return tf.assign_add(thetas, lr*grads)
