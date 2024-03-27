# set fps
# combine mask and images together

# done 1. first load the first frame which is used for setting boundary 
# done 2. Running Xmem to generate imgs while doing detection 
# 3. Merge the masks with the orignal imgs


import detect_main
import GUI
import Xmem

if __name__ == "__main__":
    detect_main.main()


# Xmem.segment('first_frame.png', 'video.mp4')
# path = GUI.select_masks_folder()

# print(path)