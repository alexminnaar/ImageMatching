import boto3
from MatchScore import MatchScore
import os
import json
import fcntl

COOCCURRENCE_COUNTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counts/cooccurrence_freq.txt")
TITLE_COUNTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counts/title_token_freq.txt")
IMAGE_TAG_COUNTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counts/image_tag_freq.txt")

#NUM_PROCESSES = 10

sqs = boto3.resource('sqs', region_name='us-east-1')
QUEUE_NAME = "image-matching-qa"

SCORE_FILE = "/home/ubuntu/scores.txt"


def write_score(score):

    with open(SCORE_FILE,"a") as g:
        fcntl.flock(g,fcntl.LOCK_EX)
        g.write(str(score)+'\n')
        fcntl.flock(g,fcntl.LOCK_UN)
    g.close()


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

    print("starting to poll SQS")

    # poll sqs forever
    while 1:

        # receives up to 10 messages at a time
        for message in queue.receive_messages():

            #get image url and title from message
            msg_body = json.loads(message.body)
            image_url = msg_body["image_url"]
            offer_title = msg_body["offer_title"]

            #print image_url
            #print offer_title

            # get match score between image and title
            score = matcher.get_score(image_url, offer_title)
            #print "match score: %f" % score


            write_score(score)

            # delete this message from queue
            message.delete()


def main():
    sqs_polling()

    # processes = []
    #
    # for i in range(1, NUM_PROCESSES):
    #     p = Process(target=sqs_polling())
    #     p.start()
    #     processes.append(p)


if __name__ == "__main__":
    main()
