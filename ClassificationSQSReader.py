import boto3
from ImageClassifier import CustomImageClassifier
# from TitleClassifier import TitleClassifier
import multiprocessing
import json
from pymemcache.client.base import Client
import sys
import logging
import hashlib
from time import sleep

LOG_FILENAME = "sqs_polling.log"
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PREFIX = "ImageMatcherService"
SEPARATOR = 0x1e


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

    # poll sqs forever
    while 1:
        # receives up to 10 messages at a time
        for message in queue.receive_messages():

            logger.warning("Process %d: Read message: %s" % (process_id, message.body))

            # get image url and title from message
            image_url = message.body

            # get image prediction
            try:
                image_pred = image_clf.run_inference_on_image(image_url)

                logger.warning('Process %d: Prediction based on image %s with confidence %s' % (
                    process_id, image_pred[0], str(image_pred[1])))

                # write prediction to memcached
                if image_pred[1] > min_prob:
                    memcache_client.set('%s' % hashlib.md5(image_url).hexdigest(), image_pred[0])
                else:
                    memcache_client.set('%s' % hashlib.md5(image_url).hexdigest(), "prediction below threshold")

                    # logger.warning("Process %d: Sucessfully wrote to memcached" % process_id)
                    # logger.warning("Process %d: memcached key %s" % (process_id, hashlib.md5(image_url).hexdigest()))
                    # logger.warning("Process %d: value in memcached: %s" % (
                    #     process_id, memcache_client.get(hashlib.md5(image_url).hexdigest())))
            except Exception:
                logger.error("Process %d: Failed to write to memcached" % process_id, exc_info=True)
                pass
            message.delete()


def main():
    queue_name = sys.argv[1]
    memcache_endpoint = sys.argv[2]
    min_prob = float(sys.argv[3])

    # keep track of processes to restart if needed. PID => Process
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
