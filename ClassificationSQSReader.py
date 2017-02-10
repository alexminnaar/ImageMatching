import boto3
from ImageClassifier import CustomImageClassifier
from TitleClassifier import TitleClassifier
import multiprocessing
import json
from pymemcache.client.base import Client
import sys
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sqs_polling(queue_name, memcache_endpoint, process_id):
    '''
    Poll SQS queue - for each message received get match score between image and title and persist result to memcache
    '''

    # SQS client config
    sqs = boto3.resource('sqs', region_name='us-east-1')
    queue = sqs.get_queue_by_name(QueueName=queue_name)

    # Memcache config
    memcache_client = Client((memcache_endpoint, 11211))

    # create image matcher object. This loads Inception model in memory.
    image_clf = CustomImageClassifier()
    title_clf = TitleClassifier()

    # poll sqs forever
    while 1:

        # receives up to 10 messages at a time
        for message in queue.receive_messages():

            logger.debug("Process %d: Read message: %s" % (process_id, message))

            # get image url and title from message
            msg_body = json.loads(message.body)
            image_url = msg_body["image_url"]
            offer_title = msg_body["offer_title"]

            # get match score between image and title
            image_pred = image_clf.run_inference_on_image(image_url)[0]
            title_pred = title_clf.title_classify([offer_title])

            logger.info('Process %d: Prediction based on Image: %s' % (process_id,image_pred))
            logger.info('Process %d: Prediction based on Title: %s' % (process_id,title_pred))

            # if image prediction is the same as title prediction
            if image_pred == title_pred:
                try:
                    # write score to memcache
                    memcache_client.set('%s' % hashlib.md5(image_url).hexdigest(), title_pred)
                    logger.info('Process %d: Successfully saved to memcached' % process_id)
                except Exception:
                    logger.error("Process %d: Failed to write to memcached" % process_id, exc_info=True)

            message.delete()


def main():
    queue_name = sys.argv[1]
    memcache_endpoint = sys.argv[2]

    # launch 4 identical polling processes
    p1 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, 1,))
    p1.start()

    p2 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, 2,))
    p2.start()

    p3 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, 3,))
    p3.start()

    p4 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, 4,))
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()


if __name__ == "__main__":
    main()
