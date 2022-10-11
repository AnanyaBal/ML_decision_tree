import sys
import numpy as np
import csv
# import ipdb
# 
def read_data(filename):
	all_records=[]
	with open(filename) as file:
		data = csv.reader(file, delimiter="\t")
		for line in data:
			all_records.append(line)

	all_records = np.asarray(all_records)[1:,].astype(int)

	return all_records

def calc_entropy_majvote_error(data):
	np_data = np.asarray(data)
	cols = np_data.shape[1]
	rows = np_data.shape[0]
	labels = np_data[:,cols-1]

	zeros_count = len(np.where(labels==0)[0])
	ones_count = rows - zeros_count
	# print("rows/////", rows)

	# ipdb.set_trace()
	if(zeros_count==0):
		# ipdb.set_trace()
		entropy = - (ones_count/rows)*(np.log(ones_count/rows)/np.log(2))

	elif(ones_count==0):
		entropy = -(zeros_count/rows)*(np.log(zeros_count/rows)/np.log(2)) 


	else:
		entropy = -(zeros_count/rows)*(np.log(zeros_count/rows)/np.log(2)) - (ones_count/rows)*(np.log(ones_count/rows)/np.log(2))

	if (zeros_count>ones_count):
		majority_vote = zeros_count
		err = ones_count

	else:
		majority_vote = ones_count
		err = zeros_count

	error = err/rows

	return entropy, error

def main():
	input_, output_ = sys.argv[1], sys.argv[2]
	input_data = read_data(input_)

	entropy, error = calc_entropy_majvote_error(input_data)

	with open(output_,'w') as file:
		file.write('entropy: {:.6f}'.format(entropy))
		file.write('\n')
		file.write('error: {:.6f}'.format(error))

if __name__ == "__main__":
	main()