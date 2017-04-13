from kafka import KafkaConsumer
from ImageClassifier import CustomImageClassifier
import multiprocessing
from pymemcache.client.base import Client
import sys
import logging
import hashlib
from time import sleep

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def kafka_polling(kafka_topic, kafka_group_id, kafka_host, memcache_endpoint, min_prob, process_id):
    '''
    Poll kafka topic - for each message received get image classification and persist result to memcache
    '''

    logger.warning("Process %d: Beginning to poll Kafka" % process_id)

    logger.warning(
        "Process %d: kafka topic: %s, groupid: %s, host: %s" % (process_id, kafka_topic, kafka_group_id, kafka_host))

    # Kafka client config
    consumer = KafkaConsumer(kafka_topic, group_id=kafka_group_id, bootstrap_servers=[kafka_host])

    # Memcache config
    memcache_client = Client((memcache_endpoint, 11211))

    # create image matcher object. This loads Inception model in memory.
    image_clf = CustomImageClassifier()

    for message in consumer:

        image_url = message.value.decode('utf-8')

        logger.warning('Process %d: Received image url %s' % (process_id, image_url))

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
            logger.error("Process %d: Failed to write to memcached" % process_id, exc_info=True)
            memcache_client.set('%s' % hashlib.md5(image_url).hexdigest(), "prediction error")


def main():
    kafka_topic = sys.argv[1]
    kafka_host = sys.argv[2]
    kafka_group_id = sys.argv[3]
    memcache_endpoint = sys.argv[4]
    min_prob = float(sys.argv[5])

    # keep track of processes in dictionary to restart if needed i.e. {PID: Process}
    processes = {}

    num_processes = range(1, 9)

    for p_num in num_processes:
        p = multiprocessing.Process(
            target=kafka_polling, args=(kafka_topic, kafka_group_id, kafka_host, memcache_endpoint, min_prob, p_num,))
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
                replacement_p = multiprocessing.Process(target=kafka_polling,
                                                        args=(
                                                            kafka_topic, kafka_group_id, kafka_host, memcache_endpoint,
                                                            min_prob, n,))
                replacement_p.start()
                processes[n] = replacement_p

            elif p.is_alive():
                logger.warning('Process %d is still alive' % n)

            # since polling never ends, sqs_polling should never successfully exit but we add this for completeness
            elif p.exitcode == 0:
                p.join()


if __name__ == "__main__":
    main()
