import boto3
from MatchScore import MatchScore
import os
from multiprocessing import Process

COOCCURRENCE_COUNTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counts/cooccurrence_feq.txt")
TITLE_COUNTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counts/title_token_feq.txt")
IMAGE_TAG_COUNTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counts/image_tag_feq.txt")

NUM_PROCESSES = 10

sqs = boto3.resource('sqs',region_name='us-east-1')
QUEUE_NAME = ""

s3 = boto3.client('s3')
S3_BUCKET = ""


def es_indexer(match_score, key):
    pass


def sqs_polling():
    '''
    Poll SQS queue - for each message received get match score between image and title and update s3 object
    '''

    # connect to queue
    queue = sqs.get_queue_by_name(QueueName=QUEUE_NAME)

    # create image matcher object. This also loads Inception model in memory.
    matcher = MatchScore(cooccurrence_freq_file=COOCCURRENCE_COUNTS,
                         image_tag_freq_file=IMAGE_TAG_COUNTS,
                         title_token_freq_file=TITLE_COUNTS)

    # poll sqs forever
    while 1:

        # receives up to 10 messages at a time
        for message in queue.receive_messages():

            if message.message_attributes is not None:
                image_url = message.message_attributes.get('imageUrl').get('StringValue')
                offer_title = message.message_attributes.get('title').get('StringValue')
                key_name = message.message_attributes.get('key-name').get('StringValue')

                # get match score between image and title
                score = matcher.get_score(image_url, offer_title)

                # update s3 with new score field
                es_indexer(score, key_name)

            # delete this message from queue
            message.delete()


def main():
    processes = []

    for i in range(1, NUM_PROCESSES):
        p = Process(target=sqs_polling())
        p.start()
        processes.append(p)


if __name__ == "__main__":
    main()
