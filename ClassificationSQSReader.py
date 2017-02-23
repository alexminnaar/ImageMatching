import boto3
from ImageClassifier import CustomImageClassifier
# from TitleClassifier import TitleClassifier
import multiprocessing
import json
from pymemcache.client.base import Client
import sys
import logging
import hashlib

LOG_FILENAME = "sqs_polling.log"
logging.basicConfig(filename=LOG_FILENAME, level=logging.WARNING)
logger = logging.getLogger(__name__)

PREFIX = "ImageMatcherService"
SEPARATOR = 0x1e


def sqs_polling(queue_name, memcache_endpoint, min_prob, process_id):
    '''
    Poll SQS queue - for each message received get match score between image and title and persist result to memcache
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
            # msg_body = json.loads(message.body)
            image_url = message.body

            # get image prediction
            try:
                image_pred = image_clf.run_inference_on_image(image_url)

                logger.warning('Process %d: Prediction based on Image: %s with confidence %s' % (
                    process_id, image_pred[0], str(image_pred[1])))

                # write prediction to memcached
                if image_pred[1] > min_prob:
                    memcache_client.set('%s' % hashlib.md5(image_url).hexdigest(), image_pred[0])
                else:
                    memcache_client.set('%s' % hashlib.md5(image_url).hexdigest(), "prediction below threshold")

                logger.warning("Process %d: Sucessfully wrote to memcached" % process_id)
                logger.warning("Process %d: memcached key %s" % (process_id, hashlib.md5(image_url).hexdigest()))
                logger.warning("Process %d: value in memcached: %s" % (
                    process_id, memcache_client.get(hashlib.md5(image_url).hexdigest())))
            except Exception:
                logger.error("Process %d: Failed to write to memcached" % process_id, exc_info=True)
                pass
            message.delete()


def main():
    queue_name = sys.argv[1]
    memcache_endpoint = sys.argv[2]
    min_prob = float(sys.argv[3])

    # launch 4 identical polling processes
    p1 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, min_prob, 1,))
    p1.start()

    p2 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, min_prob, 2,))
    p2.start()

    p3 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, min_prob, 3,))
    p3.start()

    p4 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, min_prob, 4,))
    p4.start()

    p5 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, min_prob, 5,))
    p5.start()

    p6 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, min_prob, 6,))
    p6.start()

    p7 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, min_prob, 7,))
    p7.start()

    p8 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, min_prob, 8,))
    p8.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()


if __name__ == "__main__":
    main()
