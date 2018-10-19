from env_setting import host, user, password, db
import urllib.request
import os
import time
import pymysql


# 각각의 카테고리 마다 몇개 다운받았는지 DB에 저장
class ImageDownloader:
    def __init__(self):
        self.get_connection()
        self.sharding_no = 0

        try:
            conn = self.conn
            cursor = conn.cursor()
            sql = 'SELECT sharding_no FROM similarity_param'
            cursor.execute(sql)
            self.sharding_no = cursor.fetchone()['sharding_no']

        except:
            pass
        finally:
            cursor.close()

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

    # def update_counter(self, counter_num):
    #
    #     update_counter_sql = "UPDATE similarity_param SET sharding_no = %s"
    #     try:
    #         conn = self.conn
    #         cursor = conn.cursor()
    #
    #         cursor.execute(update_counter_sql, (str(counter_num),))
    #         conn.commit()
    #
    #     except Exception as e:
    #         print(e)
    #     finally:
    #         self.download_counter = counter_num
    #         cursor.close()

    def get_all_urls(self, size=1000):
        get_all_url_sql = 'SELECT image_idx, image_url, search_keyword FROM image_info WHERE status = 0 LIMIT %s'
        result = list()

        read_success = False
        while not read_success:
            try:
                conn = self.conn
                cursor = conn.cursor()

                cursor.execute(get_all_url_sql, (size,))

                result = cursor.fetchall()

            except Exception as e:
                print(e)
                continue
            finally:
                cursor.close()

            read_success = True

        return result

    def get_specific_urls(self, keyword, size=1000):
        get_url_sql = 'SELECT image_idx, image_url, search_keyword FROM image_info WHERE status = 0 and search_keyword = %s LIMIT %s'
        result = list()

        read_success = False
        while not read_success:
            try:
                conn = self.conn
                cursor = conn.cursor()

                cursor.execute(get_url_sql, (keyword, size))

                result = cursor.fetchall()

            except Exception as e:
                print(e)
                continue
            finally:
                cursor.close()

            read_success = True

        return result

    def add_sharding_no(self):
        update_sharding_no_sql = "UPDATE similarity_param SET sharding_no = sharding_no+1"

        update_success = False
        while not update_success:
            try:
                conn = self.conn
                cursor = conn.cursor()

                cursor.execute(update_sharding_no_sql)
                conn.commit()
            except Exception as e:
                print(e)
                continue
            finally:
                cursor.close()

    def update_download_status(self, image_idx, path):
        STATUS_ONLY_DOWNLOAD = 4

        update_img_param_sql = 'UPDATE image_info SET status = %s, file_address = %s ' \
                               'WHERE image_idx = %s'

        update_success = False
        while not update_success:
            try:
                conn = self.conn
                cursor = conn.cursor()

                # cursor.execute(update_counter_sql, (image_idx,))
                cursor.execute(update_img_param_sql, (STATUS_ONLY_DOWNLOAD, path, image_idx))
                conn.commit()
            except Exception as e:
                print(e)
                continue
            finally:
                cursor.close()

            update_success = True

    def download_images(self, url_list):

        # file path and file name to download
        for url_info in url_list:
            # num_of_dir_files = len(os.walk(path).next()[2])

            downloaded_image_idx = url_info['image_idx']
            print("download", str(downloaded_image_idx) + ".jpg ...")

            # 폴더명과 파일 이름 지정
            sharding_no = str(downloaded_image_idx // 1000) + "/"

            keyword = url_info['search_keyword']


            # 파일명은 image_idx로 지정
            filename = str(url_info['image_idx']) + ".jpg"

            path = os.getcwd() + "/download/" + "/" + sharding_no

            file_path = path + filename

            without_wd_path = "/download/" + sharding_no + filename
            # Create when directory does not exist
            if not os.path.isdir(path):
                os.makedirs(path)

            # download
            is_download_success = False
            try_count = 0

            while not is_download_success:
                try:
                    # download img using url
                    urllib.request.urlretrieve(url_info['image_url'], file_path)
                except Exception as e:
                    print(e)
                    # 5회 다운로드 시도 후 실패하면 다음 이미지로 넘어감
                    if try_count < 5:
                        print("download failed. try again...")
                        continue
                    else:
                        break

                is_download_success = True
            # self.update_counter(downloaded_image_idx)
            self.update_download_status(downloaded_image_idx, path=without_wd_path)

    def run_download(self, keyword="all", size=1000):

        while True:
            start_time = time.time()
            if keyword == "all":
                # 100개씩 가져와서 무한for문 돌리기. 0이면 sleep n분
                url_list = self.get_all_urls(size)
            else:
                # 특정 키워드 가져오기
                # 100개씩 가져와서 무한for문 돌리기. 0이면 sleep n분
                url_list = self.get_specific_urls(keyword, size)

            if len(url_list) == 0:
                print('no url exists')
                # time.sleep(60*60)
                # continue
                break

            print("url list size : ", len(url_list))
            self.download_images(url_list)
            print("download 1000 images took %s seconds" % (time.time() - start_time))

        print("download finish")


if __name__ == "__main__":
    obj = ImageDownloader()
    # 실제 구동할 땐 similarity_param 테이블 초기화 시키고 할 것
    obj.run_download()
