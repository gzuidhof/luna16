import glob
import pandas as pd


# !!!!!!!!!!!!!!!!!!!!!!  IMPORTANT  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# this code is made for submission files, which have a index column.
# These columns are removed in this code.
# If your submission file hasn't a index column
# comment out both line 20 and line 26

# This function merges submission files. If files contain the same seriesuid, coordX, coordY, coordZ combination the probability are ensembled
# by taking the everage. The order of the nodules in the submission files doesn't  matter either if the files contain the same nodules.
# All the submission files have to be added to a folder called submission, all the files in this folder are ensembled.
def mergeFiles(submission_path):
	# create a list of all the submission_paths in the folder "submission"
	submission_paths = glob.glob(submission_path + '\\submission\\*.csv')

	# take the first submission file and set this in a dataFrame
	mergedSubmission = pd.read_csv(submission_paths[0])
	mergedSubmission['coordX'] = mergedSubmission['coordX'].round(2)
	mergedSubmission['coordY'] = mergedSubmission['coordY'].round(2)
	mergedSubmission['coordZ'] = mergedSubmission['coordZ'].round(2)
	# remove the index column form the submission file
	mergedSubmission.drop(mergedSubmission.columns[[0]], axis = 1, inplace = True)

	# loop over all submission files in the folder 'submission' and add them to the dataframe, except form the first one
	for index in range(1, len(submission_paths)):
		# create dataframe of submission file
		submission = pd.read_csv(submission_paths[index])
		submission['coordX'] = submission['coordX'].round(2)
		submission['coordY'] = submission['coordY'].round(2)
		submission['coordZ'] = submission['coordZ'].round(2)
		# remove the index column form the submission file
		submission.drop(submission.columns[[0]], axis = 1, inplace = True)
		# merge rows which have the same values in the colomns: seriesuid, coordX, coordY, and coordZ
		mergedSubmission = pd.merge(mergedSubmission, submission, on=['seriesuid', 'coordX', 'coordY', 'coordZ'], how = 'outer')

	# calculate the probabiliy of the merged rows and add this to the final data Frame
	print mergedSubmission
	#probability = mergedSubmission[mergedSubmission.columns[range(4, 4 + len(submission_paths))]].mean(axis = 1)
	#mergedSubmission.drop(mergedSubmission.columns[range(4, 4 + len(submission_paths))], axis = 1, inplace = True)
	probability = mergedSubmission[mergedSubmission.columns[slice(4, None)]].mean(axis = 1)
	mergedSubmission.drop(mergedSubmission.columns[slice(4,None)], axis = 1, inplace = True)
	mergedSubmission['probability'] = probability

	mergedSubmission = mergedSubmission.fillna(0)

	print mergedSubmission['probability'].count()
	print len(mergedSubmission['probability'])

	return mergedSubmission


if __name__ == "__main__":
    submission_path_first_part = '.\\csvfiles'
    file = mergeFiles(submission_path_first_part)
    file.to_csv('ensemble.csv',columns=['seriesuid','coordX','coordY','coordZ','probability'])
