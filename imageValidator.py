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
        # finally:
        #     conn.close()

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

    def init_params(self):

        try:
            conn = self.conn
            sql = 'UPDATE similarity_param SET positive_img_count = 0, negative_img_count = 0, total_img_count = 0'
            with conn.cursor() as cursor:
                cursor.execute(sql)

        except:
            print("init_params occurs exception")

        # finally:
        #     conn.close()

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

        import time
        tf.logging.set_verbosity(tf.logging.ERROR)

        # Load bytes of image files
        image_bytes = [tf.gfile.GFile(target_img_path, 'rb').read(), tf.gfile.GFile(input_path, 'rb').read()]

        hub_module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"  # 224x224

        with tf.Graph().as_default():
            input_byte, similarity_op = build_graph(hub_module_url, target_img_path)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                t0 = time.time()  # for time check

                # Inference similarities
                similarities = sess.run(similarity_op, feed_dict={input_byte: image_bytes})

                print("%d images inference time: %.2f s" % (len(similarities), time.time() - t0))

        print("- similarity: %.2f" % similarities[1])

        return similarities[1]

    def validate_img(self, threshold, size=1000):

        image_list = list()

        try:
            # get db connection
            connection = self.conn

            with connection.cursor() as cursor:
                get_image_info_sql = 'SELECT image_idx, image_url, file_address, search_keyword FROM image_info ' \
                                     'WHERE status = 4 LIMIT %s'
                cursor.execute(get_image_info_sql, (size,))
                image_list = cursor.fetchall()

            for image in image_list:
                similarity = self.similarity_test(keyword=image['search_keyword'], input_path=image['file_address'])
                if isinstance(similarity, numpy.generic):
                    similarity = numpy.asscalar(similarity)

                if similarity >= threshold:
                    status = 1
                    self.positive_img_count += 1
                    pass
                if similarity < threshold:
                    status = 2
                    self.negative_img_count += 1
                    pass

                self.total_img_count += 1

                with connection.cursor() as cursor:
                    update_img_validation_sql = 'UPDATE image_info SET status = %s, similarity = %s ' \
                                           'WHERE image_idx = %s'
                    cursor.execute(update_img_validation_sql, (status, similarity, image['image_idx']))

                    params = (self.positive_img_count, self.negative_img_count, self.total_img_count)
                    update_param_sql = 'UPDATE similarity_param SET positive_img_count = %s, negative_img_count = %s,' \
                                       ' total_img_count = %s'

                    cursor.execute(update_param_sql, params)
                    connection.commit()
        except Exception as e:
            print(e)
