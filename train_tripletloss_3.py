import facenet
import sys
import argparse
import numpy as np
import importlib
from datetime import datetime
import tensorflow as tf
import os
import cv2
import time
import itertools
import gc

epoch_list=[]
alpha_list=[]
def main(args):
    network = importlib.import_module(args.model_def)
    subdir = 'Finger'
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    train_set = facenet.get_dataset(args.data_dir)
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_placeholder = tf.placeholder(tf.float32, shape=(None,150,4), name='image')

        dynamic_alpha_placeholder = tf.placeholder(tf.float32, shape=(), name='dynamic_alpha_placeholder')

        prelogits, _ = network.inference(image_placeholder, args.keep_probability,
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
            weight_decay=args.weight_decay)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,args.embedding_size]), 3, 1)
        triplet_loss = facenet.triplet_loss(anchor, positive, negative,dynamic_alpha_placeholder)

        # learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step, args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        learning_rate = learning_rate_placeholder
        tf.summary.scalar('learning_rate', learning_rate)
        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(extra_ops):
            if args.optimizer=='ADAGRAD':
                opt = tf.train.AdagradOptimizer(learning_rate)
            elif args.optimizer=='ADADELTA':
                opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
            elif args.optimizer=='ADAM':
                opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
            elif args.optimizer=='RMSPROP':
                opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
            elif args.optimizer=='MOM':
                opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
            else:
                raise ValueError('Invalid optimization algorithm')

            train_op = opt.minimize(total_loss)





        # train_op = facenet.train(total_loss, global_step, args.optimizer,
            # learning_rate, args.moving_average_decay, tf.global_variables())
        # Create a saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            # saver.restore(sess,'./models/siamese/Finger/model-Finger.ckpt-548')
            # saver.save(sess, 'c3d_lstm/')
            epoch = 1
            final_loss=0.0
            count=0.0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                # epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, train_set, epoch, image_placeholder, learning_rate_placeholder, phase_train_placeholder,  global_step,
                    embeddings, total_loss, train_op, summary_op, summary_writer,
                    args.embedding_size, anchor, positive, negative, triplet_loss,dynamic_alpha_placeholder,final_loss,count)
                epoch+=1
                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, epoch)

    global epoch_list
    global alpha_list
    # plt.plot(epoch_list,alpha_list)
    # plt.show()
    # plt.savefig('alpha-vs-epoch.png')
    file_for_alpha.write(epoch_list)
    file_for_alpha.write(alpha_list)
    return model_dir


def train(args, sess, dataset, epoch, image_placeholder, learning_rate_placeholder, phase_train_placeholder, global_step,
          embeddings, loss, train_op, summary_op, summary_writer,
          embedding_size, anchor, positive, negative, triplet_loss,dynamic_alpha_placeholder,final_loss,count):
    batch_number = 0
    global epoch_list
    global alpha_list
    step = 0
    lr = args.learning_rate
    counter_for_margin=0
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people(dataset, args.people_per_batch, args.images_per_person)
        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        input_images = load_data(image_paths)
        emb_array = []
        for i in range(len(image_paths)):
            emb = sess.run(embeddings, feed_dict={learning_rate_placeholder: lr, phase_train_placeholder: False, image_placeholder: input_images[i:i+1,:]})
            
            emb_array.append(emb.reshape((embedding_size,)))
        emb_array=np.asarray(emb_array)
        print('%.3f' % (time.time()-start_time))
        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class,
            image_paths, args.people_per_batch, args.alpha)
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
            (nrof_random_negs, nrof_triplets, selection_time))
        print("Number of triplets "+str(nrof_triplets))
        # exit()
        if(nrof_triplets<50):
            counter_for_margin+=1
        if(counter_for_margin==3):
            if args.alpha<1.0:
                alpha_list.append(args.alpha)
                epoch_list.append(epoch)
                #epoch_list.append()
                args.alpha+=0.05
                counter_for_margin=0
        print (args.alpha)
        # Perform training on the selected triplets
        nrof_batches = int(np.ceil(nrof_triplets*3/args.batch_size))
        triplet_paths = list(itertools.chain(*triplets))
        # input_images = load_data(triplet_paths)
        train_time = 0
        i = 0
        step = 0
        summary = tf.Summary()
        loss_array = np.zeros((int(len(triplet_paths)/3),))
        nrof_batches = int(np.ceil(len(triplet_paths)/args.batch_size))
        
        for i in range(nrof_batches):
            # counter_for_margin=0
            start_time = time.time()
            current_input = load_data(triplet_paths[i*args.batch_size:i*args.batch_size+args.batch_size])
            # current_input = input_images[i*args.batch_size:i*args.batch_size+args.batch_size,:]
            feed_dict = { learning_rate_placeholder: lr, phase_train_placeholder: True,dynamic_alpha_placeholder:args.alpha, image_placeholder: current_input}
            err, _, step,emb = sess.run([loss, train_op, global_step,embeddings], feed_dict=feed_dict)
            loss_array[i] = err
            duration = time.time() - start_time
            # print(emb[0,:])
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %f' %
                  (epoch, batch_number+1, args.epoch_size, duration, err))
            final_loss+=err
            count+=1
            if count==25:
                with open('training.log','a') as f:
                    f.write(str(final_loss/count)+'\n')
                final_loss=0.0
                count=0
            # print('alpha= '+str(args.alpha))
            batch_number+=1
            train_time += duration
            summary.value.add(tag='loss', simple_value=err)
            summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
    return step


def load_data(image_paths):
    no_of_images = len(image_paths)
    images = []
    for i in range(no_of_images):
        current_image = np.load(image_paths[i])
        # current_image = cv2.resize(current_file,(450,300))
        # current_image = current_image/255.
        images.append(current_image)
    images = np.asarray(images)
    return images


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)




def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1

    return image_paths, num_per_class


def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    final_counter_sax=0
    for i in range(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in range(j, nrof_images): # For every possible positive pair.
                final_counter_sax+=1
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                # all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1


        emb_start_idx += nrof_images

    np.random.shuffle(triplets)

    return triplets, num_trips, len(triplets)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_base_dir', type=str,
        help='Directory where to write event logs.', default='./logs/siamese')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='./models/siamese')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='data/TRAIN')
    parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.6)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=0.8)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=10)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=0.96)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=22)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=1000)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=10)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=150) #90
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.siamese_acc')
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=1e-4)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
