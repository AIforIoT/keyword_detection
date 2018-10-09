import numpy
import glob

def max_pool(feature_map, size=2, stride=2):
	#Preparing the output of the pooling operation.  
    pool_out = numpy.zeros((numpy.uint16((feature_map.shape[0]-size+1)/stride),  
                            numpy.uint16((feature_map.shape[1]-size+1)/stride),  
                            feature_map.shape[-1]))  
    for map_num in range(feature_map.shape[-1]):  
        r2 = 0  
        for r in numpy.arange(0,feature_map.shape[0]-size-1, stride):  
            c2 = 0  
            for c in numpy.arange(0, feature_map.shape[1]-size-1, stride):  
                pool_out[r2, c2, map_num] = numpy.max(feature_map[r:r+size,  c:c+size])  
                c2 = c2 + 1  
            r2 = r2 +1
    return pool_out 

def conv(mat, conv_filter):
	if len(mat.shape) > 2 or len(conv_filter.shape) > 3: # Check if number of image channels matches the filter depth.  
        if mat.shape[-1] != conv_filter.shape[-1]:  
            print("Error: Number of channels in both image and filter must match.")  
            sys.exit()  
    if conv_filter.shape[1] != conv_filter.shape[2]: # Check if filter dimensions are equal.  
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')  
        sys.exit()  
    if conv_filter.shape[1]%2==0: # Check if filter diemnsions are odd.  
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')  
        sys.exit()  
   
    # An empty feature map to hold the output of convolving the filter(s) with the image.  
    feature_maps = numpy.zeros((mat.shape[0]-conv_filter.shape[1]+1,   
                                 mat.shape[1]-conv_filter.shape[1]+1,   
                                 conv_filter.shape[0])) 
	
	# Convolving the image by the filter(s).  
    for filter_num in range(conv_filter.shape[0]):  
        print("Filter ", filter_num + 1)  
        curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.  
        """  
        Checking if there are mutliple channels for the single filter. 
        If so, then each channel will convolve the image. 
        The result of all convolutions are summed to return a single feature map. 
        """  
        if len(curr_filter.shape) > 2:  
            conv_map = conv_(mat[:, :, 0], curr_filter[:, :, 0]) # Array holding the sum of all feature maps.  
            for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.  
                conv_map = conv_map + conv_(mat[:, :, ch_num],   
                                  curr_filter[:, :, ch_num])  
        else: # There is just a single channel in the filter.  
            conv_map = conv_(mat, curr_filter)  
        feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.

    return feature_maps # Returning all feature maps. 


def conv_(mat, conv_filter):  
    filter_size = conv_filter.shape[0]  
    result = numpy.zeros((mat.shape))  
    #Looping through the image to apply the convolution operation.  
    for r in numpy.uint16(numpy.arange(filter_size/2,   
                          mat.shape[0]-filter_size/2-2)):  
        for c in numpy.uint16(numpy.arange(filter_size/2, mat.shape[1]-filter_size/2-2)):  
            #Getting the current region to get multiplied with the filter.  
            curr_region = mat[r:r+filter_size, c:c+filter_size]  
            #Element-wise multipliplication between the current region and the filter.  
            curr_result = curr_region * conv_filter  
            conv_sum = numpy.sum(curr_result) #Summing the result of multiplication.  
            result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.  
              
    #Clipping the outliers of the result matrix.  
    final_result = result[numpy.uint16(filter_size/2):result.shape[0]-numpy.uint16(filter_size/2),   
                          numpy.uint16(filter_size/2):result.shape[1]-numpy.uint16(filter_size/2)]  
    return final_result 


def relu(feature_map):
	#Preparing the output of the ReLU activation function.  
    relu_out = numpy.zeros(feature_map.shape)  
    for map_num in range(feature_map.shape[-1]):  
        for r in numpy.arange(0,feature_map.shape[0]):  
            for c in numpy.arange(0, feature_map.shape[1]):  
                relu_out[r, c, map_num] = numpy.max(feature_map[r, c, map_num], 0)
    return relu_out

def audio2matrix(audio):
	"""
	Convert digital .wav to matrix of coeficients
	"""
	pass

def train_step(mat):
	"""
	My cnn model:
	conv2d->max_pool->relu->conv2d->max_pool->relu->dense
	"""
	pass

def main():
	"""
	get list of audios
	For each audio:
		audio3matrix
		train_step
	print validation & error of training
	for each validation_audio:
		calculate error
	print error of validation
	"""
	pass


if __name__ == '__main__':
	main()
