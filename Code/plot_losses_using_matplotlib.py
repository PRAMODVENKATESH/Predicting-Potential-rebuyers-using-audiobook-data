import matplotlib.pyplot as plt
from google.colab import files
plt.rcParams["figure.figsize"] = (20,10)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss(MSE)')
plt.ylim(0, 1)
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#images_dir = 'copy link to folder in the drive'
#plt.savefig(f"{images_dir}/name.png")##execute the above 2 lines of code if you wish to save the plots
plt.show()
