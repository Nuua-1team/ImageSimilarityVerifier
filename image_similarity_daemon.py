from image_similarity_check import build_graph
import tensorflow as tf
import pymysql
from env_setting import host, user, password, db
import urllib.request
import os
import numpy


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
            except:
                print("db connection exception occures")
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
    def download_img(self, url):

        # file path and file name to download
        path = os.getcwd() + "/download/"

        filename = "downloaded_img.jpg"

        # Create when directory does not exist
        if not os.path.isdir(path):
            os.makedirs(path)

        # download
        is_download_success = False
        try_count = 0
        while not is_download_success:
            try:
                urllib.request.urlretrieve(url, path + filename)
            except:
                print("download failed. try again...")
                continue

            is_download_success = True

    def similarity_test(self, search_keyword=''):
        # 나중에 new 뺄것
        if search_keyword == '경복궁':
            # target_img_path = 'reference/Kwanghwamun-reference.jpg'
            target_img_path = 'download/new_target_img.jpg'
        elif search_keyword == '창덕궁':
            target_img_path = 'reference/Chandeokgung-Injeongjeon-reference.jpg'
        else:
            # for test
            target_img_path = 'download/new_target_img.jpg'
        input_img_paths = 'download/downloaded_img.jpg'

        import time
        tf.logging.set_verbosity(tf.logging.ERROR)

        # Load bytes of image files
        image_bytes = [tf.gfile.GFile(target_img_path, 'rb').read(), tf.gfile.GFile(input_img_paths, 'rb').read()]

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

    def validate_img(self, threshold):

        url = ''

        try:
            # get db connection
            connection = self.conn
            # get download url from db

            with connection.cursor() as cursor:
                # Create a new record
                get_url_sql = 'SELECT image_idx, image_url, search_keyword FROM image_info WHERE status = 0 LIMIT 1'
                cursor.execute(get_url_sql)
                res = cursor.fetchone()
                url = res['image_url']
                image_idx = res['image_idx']
                search_keyword = res['search_keyword']
                print('url : ', url)
                print('search_keyword : ', search_keyword)

            # download img
            self.download_img(url)

            # validate img
            similarity = self.similarity_test()
            if isinstance(similarity, numpy.generic):
                similarity = numpy.asscalar(similarity)
            # threshold를 넘으면 1 안넘으면 2로 설정
            status = 0
            base_path = os.getcwd()
            img_path = ""
            img_name = ""

            if similarity > threshold:
                status = 1
                # 이미지가 이동할 경로 설정(유사한 이미지 경로)
                self.positive_img_count += 1
                base_positive_path = os.getcwd() + '/positive/'
                img_path = str(self.positive_img_count // 1000) + '/'
                img_name = str(self.positive_img_count % 1000) + '.jpg'
                file_path = base_positive_path + img_path
                file_address = base_positive_path + img_path + img_name

            else:
                status = 2
                # 이미지가 이동할 경로 설정(유사하지 않은 이미지 경로)
                self.negative_img_count += 1
                base_negative_path = os.getcwd() + '/negative/'
                img_path = str(self.negative_img_count // 1000) + '/'
                img_name = str(self.negative_img_count % 1000) + '.jpg'
                file_path = base_negative_path + img_path
                file_address = base_negative_path + img_path + img_name

            # 이미지를 file_address 로 이동시킴
            # 폴더 없으면 만드는 코드 작성(os,mkdir())

            print('base path : ' + base_path + '\\downloaded_img.jpg')
            print('path : ', file_address)

            download_file_path = str(base_path + '\\download\\downloaded_img.jpg').replace("\\", '/')
            after_move_path = file_address.replace("\\", '/')
            # Create when directory does not exist
            if not os.path.isdir(file_path):
                os.makedirs(file_path)

            os.rename(download_file_path, after_move_path)

            self.total_img_count += 1

            # update img
            with connection.cursor() as cursor:
                insert_img_param_sql = 'UPDATE image_info SET status = %s, similarity = %s, file_address = %s ' \
                                       'WHERE image_idx = %s'
                cursor.execute(insert_img_param_sql, (status, similarity, file_address, image_idx))
                connection.commit()

        finally:
            # 현재까지의 이미지 사진들을 db에 저장함

            params = (self.positive_img_count, self.negative_img_count, self.total_img_count)
            print('params' , params)

            with connection.cursor() as cursor:
                update_param_sql = 'UPDATE similarity_param SET positive_img_count = %s, negative_img_count = %s, total_img_count = %s'
                # insert_params_sql = 'INSERT INTO similarity_param(positive_img_count, negative_img_count, total_img_count)' \
                #                     ' value (%s, %s, %s)'
                cursor.execute(update_param_sql, params)
                connection.commit()

            # connection.close()


if __name__ == "__main__":
    obj = ImageValidator()
    for i in range(100):
        obj.validate_img(0.6)
