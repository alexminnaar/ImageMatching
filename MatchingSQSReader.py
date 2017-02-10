import boto3
from MatchScore import MatchScore
import multiprocessing
import os
import json
from pymemcache.client.base import Client
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sqs_polling(queue_name, memcache_endpoint, coocurrence_cts, image_tag_cts, title_cts, process_id):
    '''
    Poll SQS queue - for each message received get match score between image and title and persist result to memcache
    '''

    # SQS client config
    sqs = boto3.resource('sqs', region_name='us-east-1')
    queue = sqs.get_queue_by_name(QueueName=queue_name)

    # Memcache config
    memcache_client = Client((memcache_endpoint, 11211))

    # create image matcher object. This loads Inception model in memory.
    matcher = MatchScore(cooccurrence_freq_file=coocurrence_cts,
                         image_tag_freq_file=image_tag_cts,
                         title_token_freq_file=title_cts)

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
            score = matcher.get_score(image_url, offer_title)

            try:
                # write score to memcache
                memcache_client.set('%s|%s' % (image_url[-200:], offer_title.replace(' ', '-')), score)
            except Exception:
                logger.error("Process %d: Failed to write to memcache" % process_id, exc_info=True)

            message.delete()


def main():
    queue_name = sys.argv[1]
    memcache_endpoint = sys.argv[2]

    # count files
    cooccurrence_cts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counts/cooccurrence_freq.txt")
    title_cts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counts/title_token_freq.txt")
    image_label_cts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counts/image_tag_freq.txt")

    # Memcache config
    memcache_client = Client((memcache_endpoint, 11211))

    # launch 4 identical polling processes
    p1 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, cooccurrence_cts, image_label_cts, title_cts, 1,))
    p1.start()

    p2 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, cooccurrence_cts, image_label_cts, title_cts, 2,))
    p2.start()

    p3 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, cooccurrence_cts, image_label_cts, title_cts, 3,))
    p3.start()

    p4 = multiprocessing.Process(
        target=sqs_polling, args=(queue_name, memcache_endpoint, cooccurrence_cts, image_label_cts, title_cts, 4,))
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()


if __name__ == "__main__":
    main()
