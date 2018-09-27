import tensorflow as tf
import tensorflow_hub as hub

CHANNELS = 3  # number of image channels (RGB)


def build_graph(hub_module_url, target_image_path):
    # Step 1) Prepare pre-trained model for extracting image features.
    module = hub.Module(hub_module_url)
    height, width = hub.get_expected_image_size(module)

    # Copied a method of https://github.com/GoogleCloudPlatform/cloudml-samples/blob/bf0680726/flowers/trainer/model.py#L181
    # and fixed for all type images (not only jpeg)
    def decode_and_resize(image_str_tensor):
        """Decodes jpeg string, resizes it and returns a uint8 tensor."""
        image = tf.image.decode_image(image_str_tensor, channels=CHANNELS)
        # Note resize expects a batch_size, but tf_map supresses that index,
        # thus we have to expand then squeeze.  Resize returns float32 in the
        # range [0, uint8_max]
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(
            image, [height, width], align_corners=False)
        image = tf.squeeze(image, squeeze_dims=[0])
        image = tf.cast(image, dtype=tf.uint8)
        return image

    def to_img_feature(images):
        """Extract the feature of image vectors"""
        outputs = module(dict(images=images), signature="image_feature_vector", as_dict=True)
        return outputs['default']

    # Step 2) Extract image features of the target image.
    target_image_bytes = tf.gfile.GFile(target_image_path, 'rb').read()
    target_image = tf.constant(target_image_bytes, dtype=tf.string)
    target_image = decode_and_resize(target_image)
    target_image = tf.image.convert_image_dtype(target_image, dtype=tf.float32)
    target_image = tf.expand_dims(target_image, 0)
    target_image = to_img_feature(target_image)

    # Step 3) Extract image features of input images.
    input_byte = tf.placeholder(tf.string, shape=[None])
    input_image = tf.map_fn(decode_and_resize, input_byte, back_prop=False, dtype=tf.uint8)
    input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
    input_image = to_img_feature(input_image)

    # Step 4) Compare cosine_similarities of the target image and the input images.
    dot = tf.tensordot(target_image, tf.transpose(input_image), 1)
    similarity = dot / (tf.norm(target_image, axis=1) * tf.norm(input_image, axis=1))
    similarity = tf.reshape(similarity, [-1])

    return input_byte, similarity


if __name__ == "__main__":

    target_img_path = 'new_target_img.jpg'
    input_img_paths = ['new_test%d.jpg' % i for i in range(1, 9)]

    import time

    tf.logging.set_verbosity(tf.logging.ERROR)

    # Load bytes of image files
    image_bytes = [tf.gfile.GFile(name, 'rb').read()
                   for name in [target_img_path] + input_img_paths]

    #hub_module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/1"  # @param {type:"string"}
    hub_module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2" # 224x224

    with tf.Graph().as_default():
        input_byte, similarity_op = build_graph(hub_module_url, target_img_path)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            t0 = time.time()  # for time check

            # Inference similarities
            similarities = sess.run(similarity_op, feed_dict={input_byte: image_bytes})

            print("%d images inference time: %.2f s" % (len(similarities), time.time() - t0))

    # Display results
    print("# Target image")
    print("- similarity: %.2f" % similarities[0])

    print("# Input images")
    for similarity, input_img_path in zip(similarities[1:], input_img_paths):
        print("- similarity: %.2f" % similarity)




