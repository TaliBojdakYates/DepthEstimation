import cv2
import tensorflow as tf

class Data_Entry:
    def __init__(self, file):
        self.raw_rgb = file['rgb']
        self.raw_depths = file['depth']

        self.depths = np.array(self.raw_depths)
        self.image = np.array(self.raw_rgb).transpose(1, 2, 0) #turn from (3,480,640) to (480,640,3)
    
        expanded = np.expand_dims(self.depths, axis=-1)
        self.combined_data = np.concatenate((self.image, expanded), axis=-1)


    def get_images(self):
    
        original_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
    
    
        np_depths_normalized = (self.depths - np.min(self.depths)) / (np.max(self.depths) - np.min(self.depths)) 


        depth_image = np.uint8(np_depths_normalized* 255)
        depth_image_rgb = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
        depth_map = cv2.applyColorMap(depth_image_rgb , cv2.COLORMAP_MAGMA)

        cv2.imshow("Image", original_image)
        cv2.imshow("Depth Map", depth_map)
        cv2.waitKey(0)


def depth_loss(y_true, y_pred):
    huber_loss = tf.losses.huber(y_true, y_pred, delta=0.1)

    # Reshaping and permuting not necessary unless you are processing the data in a specific format
    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_true = tf.transpose(y_true, perm=[0, 2, 1, 3])
    y_pred = tf.transpose(y_pred, perm=[0, 2, 1, 3])

    # Depth gradient loss
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    gradient_loss = tf.reduce_mean(tf.abs(dy_true - dy_pred) + tf.abs(dx_true - dx_pred))

    # Structural similarity loss
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    # Combine the losses
    total_loss = huber_loss + 0.5 * gradient_loss + 0.1 * ssim_loss
    return total_loss