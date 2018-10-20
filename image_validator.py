from similarity_module import build_graph
import tensorflow as tf
import pymysql
from env_setting import host, user, password, db
import urllib.request
import os
import numpy
import time


class ImageValidator:

    def __init__(self, test_mode=False):

        self.get_connection()

        self.positive_img_count = 0
        self.negative_img_count = 0
        self.total_img_count = 0
        try:
            conn = self.conn
            self.test_mode = test_mode

            with conn.cursor() as cursor:
                param_init_sql = 'SELECT positive_img_count, negative_img_count, total_img_count ' \
                                 'FROM similarity_param'
                cursor.execute(param_init_sql)
                result = cursor.fetchone()

                self.positive_img_count = result['positive_img_count']
                self.negative_img_count = result['negative_img_count']
                self.total_img_count = result['total_img_count']
        except Exception as e:
            print(e)

    def get_connection(self):
        is_conn_success = False
        while not is_conn_success:
            try:
                self.conn = pymysql.connect(host=host,
                                            user=user,
                                            password=password,
                                            db=db,
                                            charset='utf8',
                                            cursorclass=pymysql.cursors.DictCursor)
            except Exception as e:
                print("db connection exception occures")
                print(e)
                continue

            if self.conn is not None:
                is_conn_success = True

        return self.conn

    def disconnect_connection(self):
        self.conn.close()

    def __del__(self):
        self.disconnect_connection()

    # png일때랑 jpg 일때랑 고려할 것
    def download_img(self, url, path="/download/"):

        # file path and file name to download
        path = os.getcwd() + path

        filename = "downloaded_img.jpg"

        # Create when directory does not exist
        if not os.path.isdir(path):
            os.makedirs(path)

        # download
        is_download_success = False
        try_count = 0

        while not is_download_success:
            try:
                # download img using url
                urllib.request.urlretrieve(url, path + filename)
            except:
                # 5회 다운로드 시도 후 실패하면 다음 이미지로 넘어감
                if try_count < 5:
                    print("download failed. try again...")
                    continue
                else:
                    break
            is_download_success = True

        return is_download_success

    # db에서 경로 입력받기(파라미터로)
    def similarity_test(self, keyword='', input_path=''):

        input_path = os.getcwd() + input_path
        person_img_path = "reference/person_img.jpg"
        hub_module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"  # 224x224

        tf.logging.set_verbosity(tf.logging.ERROR)
        # 사람과의 유사도를 먼저 측정한다.
        # Load bytes of image files
        image_bytes = [tf.gfile.GFile(person_img_path, 'rb').read(), tf.gfile.GFile(input_path, 'rb').read()]

        with tf.Graph().as_default():
            input_byte, similarity_op = build_graph(hub_module_url, person_img_path)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                t0 = time.time()  # for time check

                # Inference similarities
                similarities = sess.run(similarity_op, feed_dict={input_byte: image_bytes})

                print("%d images inference time: %.2f s" % (len(similarities), time.time() - t0))

                person_similarity = similarities[1]

        print("- person img similarity: %.2f" % similarities[1])

        if isinstance(person_similarity, numpy.generic):
            person_similarity = numpy.asscalar(person_similarity)

        # return whether similar with person or not
        if person_similarity >= 0.6:
            # [사람과 유사도가 0.6 이상이면 True, 아니면 False, 유사도]
            return [True, person_similarity]

        similarities = None
        image_bytes = None

        if keyword == "경복궁":
            target_img_path = "reference/gyungbokgung.jpg"
        elif keyword == "창덕궁":
            target_img_path = "reference/changdukgung.jpg"
        elif keyword == "광화문":
            target_img_path = "reference/gwanghwamun.jpg"
        elif keyword == "덕수궁":
            target_img_path = "reference/deoksugung.jpg"
        elif keyword == "종묘":
            target_img_path = "reference/jongmyo.jpg"
        elif keyword == "숭례문":
            target_img_path = "reference/sungnyemun.jpg"
        elif keyword == "동대문":
            target_img_path = "reference/dongdaemun.jpg"
        elif keyword == "경희궁":
            target_img_path = "reference/gyeonghuigung.jpg"
        elif keyword == "보신각":
            target_img_path = "reference/bosingak.jpg"
        else:
            # set default image changdukgung
            target_img_path = "reference/default.jpg"

        # 레퍼런스 이미지와 비교
        # Load bytes of image files
        image_bytes = [tf.gfile.GFile(target_img_path, 'rb').read(), tf.gfile.GFile(input_path, 'rb').read()]

        with tf.Graph().as_default():
            input_byte, similarity_op = build_graph(hub_module_url, target_img_path)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                t0 = time.time()  # for time check

                # Inference similarities
                similarities = sess.run(similarity_op, feed_dict={input_byte: image_bytes})

                print("%d images inference time: %.2f s" % (len(similarities), time.time() - t0))

                arch_similarity = similarities[1]

        print("- similarity: %.2f" % arch_similarity)

        if isinstance(arch_similarity, numpy.generic):
            arch_similarity = numpy.asscalar(arch_similarity)
        # [사람과 유사하면 True 아니면 False, 유사도]
        return [False, arch_similarity]

    def validate_img(self, threshold, size=1000):
        STATUS_POSITIVE = 1
        STATUS_NEGATIVE = 2
        STATUS_PERSON = 2

        try:
            # get db connection
            connection = self.conn
            while True:
                image_list = list()
                with connection.cursor() as cursor:
                    get_image_info_sql = 'SELECT image_idx, image_url, file_address, search_keyword FROM image_info ' \
                                         'WHERE status = 4 LIMIT %s'
                    cursor.execute(get_image_info_sql, (size,))
                    image_list = cursor.fetchall()

                if not image_list:
                    print("no more image_list")
                    break

                for image in image_list:
                    similarity_result = self.similarity_test(keyword=image['search_keyword'],
                                                             input_path=image['file_address'])

                    is_similar_with_people = similarity_result[0]
                    similarity = similarity_result[1]

                    # [사람과 유사하면 True, 유사도]

                    if isinstance(similarity, numpy.generic):
                        similarity = numpy.asscalar(similarity)

                    # 사람과 유사한 경우
                    if is_similar_with_people:
                        status = STATUS_PERSON
                        self.negative_img_count += 1
                    # 사람과 유사하지 않고, 유사도가 역치보다 높은 경우
                    elif similarity >= threshold:
                        status = STATUS_POSITIVE
                        self.positive_img_count += 1
                    # 사람과 유사하지 않고, 유사도가 역치보다 높은 경우
                    elif similarity < threshold:
                        status = STATUS_NEGATIVE
                        self.negative_img_count += 1

                    self.total_img_count += 1

                    with connection.cursor() as cursor:
                        if is_similar_with_people:
                            update_img_validation_sql = 'UPDATE image_info SET status = %s, similarity_person = %s ' \
                                                        'WHERE image_idx = %s'
                        else:
                            update_img_validation_sql = 'UPDATE image_info SET status = %s, similarity = %s ' \
                                                        'WHERE image_idx = %s'

                        cursor.execute(update_img_validation_sql, (status, similarity, image['image_idx']))

                        params = (self.positive_img_count, self.negative_img_count, self.total_img_count)
                        update_param_sql = 'UPDATE similarity_param SET positive_img_count = %s, ' \
                                           'negative_img_count = %s, total_img_count = %s'

                        cursor.execute(update_param_sql, params)
                        connection.commit()
        except Exception as e:
            print("exception occurs during process validate_img")
            print(e)

        finally:
            print("validate image finished")
