import boto3
from ImageClassifier import CustomImageClassifier
import multiprocessing
from pymemcache.client.base import Client
import sys
import logging
import hashlib
from time import sleep

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def sqs_polling(queue_name, memcache_endpoint, min_prob, process_id):
    '''
    Poll SQS queue - for each message received get image classification and persist result to memcache
    '''

    logger.warning("Process %d: Beginning to poll SQS" % process_id)

    # SQS client config
    sqs = boto3.resource('sqs', region_name='us-east-1')
    queue = sqs.get_queue_by_name(QueueName=queue_name)

    # Memcache config
    memcache_client = Client((memcache_endpoint, 11211))

    # create image matcher object. This loads Inception model in memory.
    image_clf = CustomImageClassifier()

    no_messages = False

    # poll sqs forever
    while 1:

        # sleep longer if there are no messages on the queue the last time it was polled
        if no_messages:
            logger.warning('Process %d: no messages received so sleeping for 15 minutes' % process_id)
            sleep(900.0)
            queue = sqs.get_queue_by_name(QueueName=queue_name)

        # get next batch of messages (up to 10 at a time)
        message_batch = queue.receive_messages(MaxNumberOfMessages=10, WaitTimeSeconds=20)

        logger.warning('Process %d: received %d messages' % (process_id, len(message_batch)))

        if len(message_batch) == 0:
            no_messages = True
        else:
            no_messages = False

        # process messages
        for message in message_batch:

            # get image url from message
            image_url = message.body

            # predict image category and persist to memcache
            try:
                image_pred = image_clf.run_inference_on_image(image_url)

                logger.warning('%s | %s |%s' % (image_url, image_pred[0], str(image_pred[1])))

                # write prediction to memcached if we are confident enough
                if image_pred[1] > min_prob:
                    memcache_client.set('%s' % hashlib.md5(image_url).hexdigest(), image_pred[0])
                else:
                    memcache_client.set('%s' % hashlib.md5(image_url).hexdigest(), "prediction below threshold")

            except Exception:
                logger.error("Failed to write to memcached", exc_info=True)
                memcache_client.set('%s' % hashlib.md5(image_url).hexdigest(), "prediction error")

            # messages are always deleted
            finally:
                message.delete()


def main():
    queue_name = sys.argv[1]
    memcache_endpoint = sys.argv[2]
    min_prob = float(sys.argv[3])

    # keep track of processes in dictionary to restart if needed i.e. {PID: Process}
    processes = {}

    num_processes = range(1, 9)

    for p_num in num_processes:
        p = multiprocessing.Process(
            target=sqs_polling, args=(queue_name, memcache_endpoint, min_prob, p_num,))
        p.start()
        processes[p_num] = p

    # periodically poll child processes to check if they are still alive
    while len(processes) > 0:

        # check every 5 minutes
        sleep(300.0)

        for n in processes.keys():
            p = processes[n]

            # if process is dead, create a new one to take its place
            if not p.is_alive():
                logger.error('Process %d is dead! Starting new process to take its place.' % n)
                replacement_p = multiprocessing.Process(target=sqs_polling,
                                                        args=(queue_name, memcache_endpoint, min_prob, n,))
                replacement_p.start()
                processes[n] = replacement_p

            elif p.is_alive():
                logger.warning('Process %d is still alive' % n)

            # since polling never ends, sqs_polling should never successfully exit but we add this for completeness
            elif p.exitcode == 0:
                p.join()


if __name__ == "__main__":
    main()
