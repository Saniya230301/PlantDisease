
    op = str(list2[classes[0]])

    return render_template('plant_disease.html', op=op)
       
#     img = keras.utils.load_img(predict_dir_path + imagefile, target_size=(224, 224))
#     x = keras.utils.img_to_array(img)
#     x = np.expand_dims(x, axis=0)