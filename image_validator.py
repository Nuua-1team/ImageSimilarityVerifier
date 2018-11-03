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
        self.graph_init()

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

    def db_disconnect(self):
        self.conn.close()

    def __del__(self):
        self.db_disconnect()

    def graph_init(self):

        hub_module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"  # 224x224
        keyword_list = ['경복궁', '창덕궁', '광화문', '덕수궁', '종묘', '숭례문', '동대문', '경희궁', '보신각', '기타', '인물']
        path_list = ["reference/gyeongbokgung.jpg", "reference/changdukgung.jpg", "reference/gwanghwamun.jpg",
                     "reference/deoksugung.jpg", "reference/jongmyo.jpg", "reference/sungnyemun.jpg",
                     "reference/dongdaemun.jpg", "reference/gyeonghuigung.jpg", "reference/bosingak.jpg",
                     "reference/default.jpg", "reference/person_img.jpg"]

        self.graph_list = [[build_graph(hub_module_url, path)]for path in path_list]
        self.ref_image_list = [tf.gfile.GFile(path, 'rb').read() for path in path_list]


    # db에서 경로 입력받기(파라미터로)
    def similarity_test(self, keyword='', input_path=''):

        input_path = os.getcwd() + input_path
        graph_list = self.graph_list
        ref_image_list = self.ref_image_list

        tf.logging.set_verbosity(tf.logging.ERROR)
        # 사람과의 유사도를 먼저 측정한다.
        # Load bytes of image files
        # try:
        person_idx = 10
        image_bytes = [ref_image_list[person_idx], tf.gfile.GFile(input_path, 'rb').read()]

        with tf.Graph().as_default():
            input_byte = graph_list[person_idx][0]
            similarity_op = graph_list[person_idx][1]

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

        if keyword == "경복궁":
            target_idx = 0
        elif keyword == "창덕궁":
            target_idx = 1
        elif keyword == "광화문":
            target_idx = 2
        elif keyword == "덕수궁":
            target_idx = 3
        elif keyword == "종묘":
            target_idx = 4
        elif keyword == "숭례문":
            target_idx = 5
        elif keyword == "동대문":
            target_idx = 6
        elif keyword == "경희궁":
            target_idx = 7
        elif keyword == "보신각":
            target_idx = 8
        else:
            target_idx = 9

            # 레퍼런스 이미지와 비교
        # Load bytes of image files
        image_bytes = [ref_image_list[target_idx], tf.gfile.GFile(input_path, 'rb').read()]

        with tf.Graph().as_default():
            input_byte = graph_list[target_idx][0]
            similarity_op = graph_list[target_idx][1]

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

    def validate_img(self, threshold, size=100):
        STATUS_POSITIVE = 1
        STATUS_NEGATIVE = 2
        STATUS_PERSON = 2

        try:
            # get db connection
            connection = self.conn
            while True:
                image_list = list()
                with connection.cursor() as cursor:
                    # DB에서 다운로드가 완료된 이미지 정보와 경로를 size만큼 가져옴
                    get_image_info_sql = 'SELECT image_idx, image_url, file_address, search_keyword FROM image_info ' \
                                         'WHERE status = 4 LIMIT %s'
                    cursor.execute(get_image_info_sql, (size,))
                    image_list = cursor.fetchall()

                if not image_list:
                    print("no more image_list")
                    break

                for image in image_list:
                    # 유사도 측정 결과 크기가 2인 리스트로 반환
                    # [사람과 유사한지 여부(boolean), 유사도(float)]
                    print("img_idx : ", image['image_idx'])
                    if not os.path.exists(os.getcwd() + image['file_address']):
                        # 이미지가 존재하지 않을 경우 db에서 이 idx 지우고 다음 이미지로 넘어감
                        with connection.cursor() as cursor:
                            sql = "DELETE FROM image_info WHERE image_idx = %s"
                            cursor.execute(sql, (image['image_idx'],))
                            connection.commit()
                        continue
                        pass

                    else:
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
                            update_img_validation_sql = 'UPDATE image_info SET status = %s, similarity_person = %s, similarity = 0 ' \
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


if __name__ == "__main__":
    validator = ImageValidator()
    validator.validate_img(threshold=0.6, size=10)
