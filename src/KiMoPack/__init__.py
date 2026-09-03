import urllib3
import sys
import os
import pathlib
import shutil
from pathlib import Path

from .paths import check_folder

def download_notebooks(libraries_only=False):
	'''function loads the workflow notebooks into the active folder
	if libraries_only is set to True, only the function library and the import library are loaded'''
	http = urllib3.PoolManager()
	list_of_tools=['Function_library_overview.pdf',
					'function_library.py',
					'import_library.py',
					'TA_Advanced_Fit.ipynb',
					'TA_comparative_plotting_and_data_extraction.ipynb',
					'TA_Raw_plotting.ipynb',
					'TA_Raw_plotting_and_Simple_Fit.ipynb',
					'TA_single_scan_handling.ipynb',
					'Streak_camera_analysis.ipynb',
					'XES_Raw_plotting_and_Simple_Fit.ipynb']
	print('Now downloading the workflow tools')
	for i,f in enumerate(list_of_tools):
		url = "https://raw.githubusercontent.com/erdzeichen/KiMoPack/main/Workflow_tools/%s"%f
		print('Downloading Workflow Tools/%s'%f)
		with open(check_folder(path = 'Workflow_tools', current_path = os.getcwd(), filename = f), 'wb') as out:
			r = http.request('GET', url, preload_content=False)
			shutil.copyfileobj(r, out)
		if libraries_only and i==2:
			break
def download_all(single_tutorial=None):
	''' function loads workflow notebooks and example files and tutorials'''
	http = urllib3.PoolManager()
	if single_tutorial is None:
		download_notebooks()
		print('Now downloading the workflow tools and tutorials')
	else:
		download_notebooks(libraries_only=True)
		print('Libraries downloaded')
	list_of_example_data=['sample_1_chirp.dat',
							'Sample_2_chirp.dat',
							'sample_1.hdf5',
							'sample_2.hdf5',
							'Sample_1.SIA',
							'Sample_2.SIA',
							'XES_diff.SIA',
							'XES_on.SIA',
							'FeCM02-266nm-4mw-QB390-t6-G63-w450-s150-556ms-E100.dat',
							'FeCM02-266nm-4mw-QB390-t6-G63-w450-s150-556ms-E100_chirp.dat']
	print('Now downloading the example files')
	if (single_tutorial is None) or (single_tutorial == 'workflow'): #we do not use this to download data for Colab 
		for f in list_of_example_data:
			url = "https://raw.githubusercontent.com/erdzeichen/KiMoPack/main/Workflow_tools/Data/%s"%f
			print('Downloading Workflow Tools/Data/%s'%f)
			with open(check_folder(path = 'Workflow_tools'+os.sep+'Data', current_path = os.getcwd(), filename = f), 'wb') as out:
				r = http.request('GET', url, preload_content=False)
				shutil.copyfileobj(r, out)
	
	list_of_tutorials=['function_library.py',
						'Function_library_overview.pdf',
						'import_library.py',
						'KiMoPack_tutorial_0_Introduction.ipynb',
						'KiMoPack_tutorial_1_Fitting.ipynb',
						'KiMoPack_tutorial_2_Fitting.ipynb',
						'KiMoPack_tutorial_3_CompareFit.ipynb',
						'KiMoPack_tutorial_4_ScanHandling.ipynb',
						'KiMoPack_tutorial_5_MultiModal.ipynb',
						'KiMoPack_tutorial_6_Advanced_Fitting_options.ipynb']
	if single_tutorial is None: #we do not use this to download data for Colab 
		print('Now downloading tutorials')
		for f in list_of_tutorials:
			url = "https://raw.githubusercontent.com/erdzeichen/KiMoPack/main/Tutorial_Notebooks/%s"%f
			print('Downloading tutorial %s'%f)
			with open(check_folder(path = 'Tutorial_Notebooks', current_path = os.getcwd(), filename = f), 'wb') as out:
				r = http.request('GET', url, preload_content=False)
				shutil.copyfileobj(r, out)
	tutorial_data={'Compare':['TA_Ru-dppz_400nm_DCM_paral.hdf5','TA_Ru-dppz_400nm_H2O_paral.hdf5','UVvis_SEC_Rudppz_ACN.dat'],
				   'Master':['TA_Ru-dppz_400nm_ACN_paral.hdf5'],
				   'Fitting-1':['TA_Ru-dppz_400nm_ACN.SIA','TA_Ru-dppz_400nm_ACN_chirp.dat','TA_Ru-dppz_400nm_DCM.SIA','TA_Ru-dppz_400nm_DCM_chirp.dat','TA_Ru-dppz_400nm_H2O.SIA','TA_Ru-dppz_400nm_H2O_chirp.dat'],
					'Fitting-2':['TA_Ru-dppz_400nm_ACN.SIA','TA_Ru-dppz_400nm_ACN_chirp.dat'],
					'Introduction':['catalysis1.SIA','catalysis2.SIA','con_1.SIA','con_1_solved.hdf5','con_2.SIA','con_2_chirp.dat','con_3.SIA','con_4.SIA','con_5.SIA','con_6.SIA','con_6_chirp.dat','full_consecutive_fit.hdf5','full_consecutive_fit_with_GS.hdf5','sample_1_chirp.dat'],
					'Scan':['ACN_001.SIA','ACN_002.SIA','ACN_003.SIA','ACN_004.SIA','ACN_005.SIA','ACN_006.SIA','ACN_007.SIA','ACN_008.SIA','ACN_009.SIA','TA_Ru-dppz_400nm_ACN_mean.SIA','TA_Ru-dppz_400nm_ACN_mean_chirp.dat'],
					'MultiModal':['combined_optical_spectrum.SIA','XES_on.SIA']}
	url = "https://raw.githubusercontent.com/erdzeichen/KiMoPack/main/Tutorial_Notebooks/Data"
	for key in tutorial_data.keys():
		if single_tutorial is not None:   #this is a shortcut to download data fror Colab use
			if not key==single_tutorial:
				continue
		for f in tutorial_data[key]:
			if 'Master' in key:
				url = "https://raw.githubusercontent.com/erdzeichen/KiMoPack/main/Tutorial_Notebooks/Data/Compare/Master/%s"%f
				with open(check_folder(path = os.sep.join(['Tutorial_Notebooks','Data','Compare','Master']), current_path = os.getcwd(), filename = f), 'wb') as out:
					r = http.request('GET', url, preload_content=False)
					shutil.copyfileobj(r, out)
			else:
				url = "https://raw.githubusercontent.com/erdzeichen/KiMoPack/main/Tutorial_Notebooks/Data/%s/%s"%(key,f)
				with open(check_folder(path = os.sep.join(['Tutorial_Notebooks','Data',key]), current_path = os.getcwd(), filename = f), 'wb') as out:
					r = http.request('GET', url, preload_content=False)
					shutil.copyfileobj(r, out)
	tutorial_images=['Cor_Chirp.gif','Fig1_parallel_model.png','Fig2_consecutive_model.png','Fig3_complex_model.png','Intro_tutorial.png','Model_selection.jpg']
	if single_tutorial is None:
		for f in tutorial_images:
			url = "https://raw.githubusercontent.com/erdzeichen/KiMoPack/main/Tutorial_Notebooks/img/%s"%f
			with open(check_folder(path = os.sep.join(['Tutorial_Notebooks','img']), current_path = os.getcwd(), filename = f), 'wb') as out:
				r = http.request('GET', url, preload_content=False)
				shutil.copyfileobj(r, out)
