from image_similarity_check import build_graph
import tensorflow as tf
import pymysql
from env_setting import host, user, password, db
import urllib.request
import os


class ImageValidator:

    def __init__(self, test_mode=False):
        try:
            conn = self.get_connection()
            self.test_mode = test_mode

            with conn.cursor() as cursor:
                param_init_sql = 'SELECT positive_img_count, negative_img_count, total_img_count ' \
                      'FROM similarity_param'
                cursor.execute(param_init_sql)
                result = cursor.fetchone()

                self.positive_img_count = result['positive_img_count']
                self.negative_img_count = result['negative_img_count']
                self.total_img_count = result['total_img_count']
        finally:
            conn.close()

    def get_connection(self):
        conn = pymysql.connect(host=host,
                               user=user,
                               password=password,
                               db=db,
                               charset='utf8',
                               cursorclass=pymysql.cursors.DictCursor)

        return conn

    # png일때랑 jpg 일때랑 고려할 것
    def download_img(self, url):

        # file path and file name to download
        path = os.getcwd() + "/download"

        filename = "downloaded_img.jpg"

        # Create when directory does not exist
        if not os.path.isdir(path):
            os.makedirs(path)

        # download
        urllib.request.urlretrieve(url, path + filename)

    def similarity_test(self, search_keyword=''):
        # 나중에 new 뺄것
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
            connection = self.get_connection()
            # get download url from db

            with connection.cursor() as cursor:
                # Create a new record
                get_url_sql = 'SELECT image_url, search_keyword FROM image_info WHERE image_info.is_saved = 0 LIMIT 1'
                cursor.execute(get_url_sql)
                res = cursor.fetchone()
                url = res['image_url']
                search_keyword = res['search_keyword']
                print('url : ', url)
                print('search_keyword : ', search_keyword)

            # download img
            self.download_img(url)

            # validate img
            similarity = self.similarity_test()
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

            else:
                status = 2
                # 이미지가 이동할 경로 설정(유사하지 않은 이미지 경로)
                self.negative_img_count += 1
                base_positive_path = os.getcwd() + '/negative/'
                img_path = str(self.negative_img_count // 1000) + '/'
                img_name = str(self.negative_img_count % 1000) + '.jpg'

            file_address = base_path + img_path + img_name

            # 이미지를 file_address 로 이동시킴
            # 폴더 없으면 만드는 코드 작성(os,mkdir())
            if not os.path.isdir(base_path+img_path):
                os.makedirs(base_path+img_path)

            os.rename(base_path + '/downloaded_img.jpg', file_address)

            self.total_img_count += 1

            # update img
            with connection.cursor() as cursor:
                insert_img_param_sql = 'UPDATE image_info SET status = %s, similarity = %s, file_address = %s'
                cursor.execute(insert_img_param_sql, (status, similarity, file_address))

        finally:
            # 현재까지의 이미지 사진들을 db에 저장함

            counts = dict()
            counts['positive_img_count'] = self.positive_img_count
            counts['negative_img_count'] = self.negative_img_count
            counts['total_img_count'] = self.total_img_count

            params = (self.positive_img_count, self.negative_img_count, self.total_img_count)

            with connection.cursor() as cursor:
                insert_params_sql = 'INSERT INTO similarity_param(positive_img_count, negative_img_count, total_img_count)' \
                                    ' value (%s, %s, %s)'
                cursor.execute(insert_params_sql, params)

            connection.close()


if __name__ == "__main__":
    obj = ImageValidator()
