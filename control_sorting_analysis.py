from joblib import Parallel, delayed
import datetime
import glob
import os
import multiprocessing
import shutil
import subprocess
import sys
import traceback
import time
import Logger
from PreClustering import pre_process_ephys_data
from PostSorting import post_process_sorted_data
from PostSorting import post_process_sorted_data_vr
from PostSorting import post_process_sorted_data_sleep
from PostSorting import post_process_sorted_data_opto

# set this to true if you want to skip the spike sorting step and use ths data from the server
skip_sorting = False

mountainsort_tmp_folder = '/tmp/mountainlab/'
sorting_folder = '/home/nolanlab/to_sort/recordings/'
to_sort_folder = '/home/nolanlab/to_sort/'
if os.environ.get('SERVER_PATH_FIRST_HALF'):
    server_path_first_half = os.environ['SERVER_PATH_FIRST_HALF']
    print(f'Using a custom server path: {server_path_first_half}')
else:
    # server_path_first_half = '/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/'
    server_path_first_half = '/mnt/datastore/'

#server_path_first_half = '/home/nolanlab/ardbeg/'
matlab_params_file_path = '/home/nolanlab/PycharmProjects/in_vivo_ephys_openephys/PostClustering/'
downtime_lists_path = '/home/nolanlab/to_sort/sort_downtime/'


def check_folder(sorting_path):
    recording_to_sort = False
    for dir, sub_dirs, files in os.walk(sorting_path):
        if not sub_dirs and not files:
            return recording_to_sort
        if not files:
            print('I am looking here: ', dir, sub_dirs)

        else:
            print('I found something, and I will try to sort it now.')
            recording_to_sort = find_sorting_directory()
            if recording_to_sort is False:
                return recording_to_sort
            else:
                return recording_to_sort


def find_sorting_directory():
    for name in glob.glob(sorting_folder + '*'):
        os.path.isdir(name)
        if check_if_recording_was_copied(name) is True:
            return name
        else:
            print('This recording was not copied completely, I cannot find copied.txt')
    return False


def check_if_recording_was_copied(recording_to_sort):
    if os.path.isfile(recording_to_sort + '/copied.txt') is True:
        return True
    else:
        return False


# return whether it is vr or openfield
def get_session_type(recording_directory):
    session_type = 'undefined'
    parameters_path = recording_directory + '/parameters.txt'
    try:
        param_file_reader = open(parameters_path, 'r')
        parameters = param_file_reader.readlines()
        parameters = list([x.strip() for x in parameters])
        session_type = parameters[0]

        if session_type == 'vr':
            print('This is a VR session.')
        elif session_type == 'openfield':
            print('This is an open field session')
        elif session_type == 'sleep':
            print('This is a sleep session')
        elif session_type == 'opto':
            print('This is an opto-tagging session')
        else:
            print('Session type is not specified. '
                  'You need to write vr/openfield/sleep/opto in the first line of the parameters.txt file. '
                  'You put {} there.'.format(session_type))
    except Exception as ex:
        print('There is a problem with the parameter file.')
        print(ex)
    return session_type


def get_location_on_server(recording_directory):
    parameters_path = recording_directory + '/parameters.txt'
    param_file_reader = open(parameters_path, 'r')
    parameters = param_file_reader.readlines()
    parameters = list([x.strip() for x in parameters])
    location_on_server = parameters[1]
    return location_on_server


def get_tags_parameter_file(recording_directory):
    tags = False
    parameters_path = recording_directory + '/parameters.txt'
    param_file_reader = open(parameters_path, 'r')
    parameters = param_file_reader.readlines()
    parameters = list([x.strip() for x in parameters])
    if len(parameters) > 2:
        tags = parameters[2]
    return tags


def check_for_paired(running_parameter_tags):
    paired_recordings = None
    if running_parameter_tags is not False:
        tags = [x.strip() for x in running_parameter_tags.split('*')]
        for tag in tags:
            if tag.startswith('paired'):
                paired_recordings = str(tag.split("=")[1]).split(',')
    return paired_recordings


# write file 'crash_list.txt' in top level dir with list of recordings that could not be sorted
def add_to_list_of_failed_sortings(recording_to_sort):
    if os.path.isfile(to_sort_folder + "/crash_list.txt") is False:
        crash_writer = open(to_sort_folder + 'crash_list.txt', 'w', newline='\n')

    else:
        crash_writer = open(to_sort_folder + '/crash_list.txt', 'a', newline='\n')
    crashed_recording = str(recording_to_sort) + '\n'
    crash_writer.write(crashed_recording)
    crash_writer.close()


def remove_folder_from_server_and_copy(recording_to_sort, location_on_server, name):
    if os.path.exists(server_path_first_half + location_on_server + name) is True:
        shutil.rmtree(server_path_first_half + location_on_server + name)
    try:
        if os.path.exists(recording_to_sort + name) is True:
            shutil.copytree(recording_to_sort + name, server_path_first_half + location_on_server + name)
    except shutil.Error as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        print('I am letting this exception pass, because shutil.copytree seems to have some permission issues '
              'I could not resolve, but the files are actually copied successfully.')
        pass


def delete_ephys_for_recording(recording):
    if os.path.exists(recording + "/Electrophysiology") is True:
        shutil.rmtree(recording + "/Electrophysiology")


def copy_output_to_server(recording_to_sort, location_on_server):
    remove_folder_from_server_and_copy(recording_to_sort, location_on_server, '/Figures')
    remove_folder_from_server_and_copy(recording_to_sort, location_on_server, '/Firing_fields')
    remove_folder_from_server_and_copy(recording_to_sort, location_on_server, '/MountainSort')


def call_post_sorting_for_session_type(recording_to_sort, session_type, tags):
    if session_type == "openfield":
        post_process_sorted_data.post_process_recording(recording_to_sort, 'openfield', running_parameter_tags=tags)
    elif session_type == "vr":
        post_process_sorted_data_vr.post_process_recording(recording_to_sort, 'vr', running_parameter_tags=tags)
    elif session_type == "sleep":
        post_process_sorted_data_sleep.post_process_recording(recording_to_sort, 'sleep', running_parameter_tags=tags)
    elif session_type == "opto":
        post_process_sorted_data_opto.post_process_recording(recording_to_sort, 'opto', running_parameter_tags=tags)


def run_post_sorting_for_all_recordings(recording_to_sort, session_type,
                                        paired_recordings_to_sort, paired_session_types,
                                        stitch_points, tags):

    recording_to_sort, recs_length = pre_process_ephys_data.split_back(recording_to_sort, stitch_points)

    call_post_sorting_for_session_type(recording_to_sort, session_type, tags)
    delete_ephys_for_recording(recording_to_sort)
    for index, paired_recording in enumerate(paired_recordings_to_sort):
        print('I will run the post-sorting scrpits for: ' + paired_recording)
        call_post_sorting_for_session_type(paired_recording, paired_session_types[index], tags)
        copy_paired_outputs_to_server(paired_recording)
        delete_ephys_for_recording(paired_recording)


def copy_paired_outputs_to_server(paired_recordings_to_sort):
    if os.path.exists(paired_recordings_to_sort) is True:
        server_loc = get_location_on_server(paired_recordings_to_sort)
        copy_output_to_server(paired_recordings_to_sort, server_loc)
        shutil.rmtree(paired_recordings_to_sort)


def call_spike_sorting_analysis_scripts(recording_to_sort, tags, paired_recording=None):
    print('I will analyze ' + recording_to_sort)
    print(datetime.datetime.now())
    try:
        session_type = get_session_type(recording_to_sort)
        location_on_server = get_location_on_server(recording_to_sort)

        sys.stdout = Logger.Logger(server_path_first_half + location_on_server + '/sorting_log.txt')

        if paired_recording is not None:
            print('Multiple recordings will be sorted together: ' + recording_to_sort + ' ' + str(paired_recording))
            if not isinstance(paired_recording, list):
                paired_recording = [paired_recording]
            paired_recordings_to_sort = []
            paired_session_types = []
            paired_locations_on_server = []
            for recording in paired_recording:
                paired_recording = copy_recording_to_sort_to_local(recording)
                paired_recording_to_sort = sorting_folder + recording.split('/')[-1]
                paired_recordings_to_sort.append(paired_recording_to_sort)
                paired_location_on_server = get_location_on_server(paired_recording_to_sort)
                paired_locations_on_server.append(paired_location_on_server)
                paired_session_type = get_session_type(paired_recording_to_sort)
                paired_session_types.append(paired_session_type)
            recording_to_sort, stitch_points = pre_process_ephys_data.stitch_recordings(recording_to_sort, paired_recordings_to_sort)
        
        if not skip_sorting:
            pre_process_ephys_data.pre_process_data(recording_to_sort)

            print('I finished pre-processing the first recording. I will call MountainSort now.')
            os.chmod('/home/nolanlab/to_sort/run_sorting.sh', 484)

            subprocess.call('/home/nolanlab/to_sort/run_sorting.sh', shell=True)
            os.remove('/home/nolanlab/to_sort/run_sorting.sh')

            print('MS is done')

        # call python post-sorting scripts
        print('Post-sorting analysis (Python version) will run now.')

        if paired_recording is not None:
            run_post_sorting_for_all_recordings(recording_to_sort, session_type,
                                              paired_recordings_to_sort, paired_session_types,
                                              stitch_points, tags)
            for path_to_paired_recording in paired_recordings_to_sort:
                if os.path.exists(path_to_paired_recording) is True:
                    shutil.rmtree(path_to_paired_recording)

        else:
           # (recording_to_sort, session_type, stitch_point, tags, recs_length, paired_order=None)
            call_post_sorting_for_session_type(recording_to_sort, session_type, tags=tags)

        if os.path.exists(recording_to_sort) is True:
            copy_output_to_server(recording_to_sort, location_on_server)

        shutil.rmtree(recording_to_sort)

        if not skip_sorting:
            shutil.rmtree(mountainsort_tmp_folder)
    
    except Exception as ex:
        print('There is a problem with this file. '
              'I will move on to the next one. This is what Python says happened:')
        print(ex)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback)
        add_to_list_of_failed_sortings(recording_to_sort)
        location_on_server = get_location_on_server(recording_to_sort)
        if os.path.exists(recording_to_sort) is True:
            copy_output_to_server(recording_to_sort, location_on_server)

        if not os.environ.get('DEBUG'):  # Keep the recording files during debug run
            shutil.rmtree(recording_to_sort)

            if paired_recording is not None:
                for path_to_paired_recording in paired_recordings_to_sort:
                    if os.path.exists(path_to_paired_recording) is True:
                        shutil.rmtree(path_to_paired_recording)

            if os.path.exists(mountainsort_tmp_folder) is True:
                shutil.rmtree(mountainsort_tmp_folder)

        if os.environ.get('SINGLE_RUN'):
            print('Single run mode was active during the error. '
                  'I will quit immediately with a nonzero exit status instead of continuing to the next recording.')
            exit(1)  # an exit status of 1 means unsuccessful termination/program failure


def delete_processed_line(list_to_read_path):
    with open(list_to_read_path, 'r') as file_in:
        data = file_in.read().splitlines(True)
    with open(list_to_read_path, 'w') as file_out:
        file_out.writelines(data[1:])


def copy_file(filename, path_local):
    if os.path.isfile(filename) is True:
        if filename.split('.')[-1] == 'txt':
            shutil.copy(filename, path_local + '/' + filename.split('/')[-1])
        if filename.split('.')[-1] == 'csv':
            shutil.copy(filename, path_local + '/' + filename.split('/')[-1])
        if filename.split('.')[-1] == 'continuous':
            shutil.copy(filename, path_local + '/' + filename.split('/')[-1])
        if filename.split('.')[-1] == 'pkl':
            shutil.copy(filename, path_local + '/' + filename.split('/')[-1])
        if filename.split('.')[-1] == 'events':
            shutil.copy(filename, path_local + '/' + filename.split('/')[-1])


def copy_recording_to_sort_to_local(recording_to_sort):
    path_server = server_path_first_half + recording_to_sort
    recording_to_sort_folder = recording_to_sort.split("/")[-1]
    path_local = sorting_folder + recording_to_sort_folder
    print('I will copy a folder from the server now. It will take a while.')
    if os.path.exists(path_server) is False:
        print('This folder does not exist on the server:')
        print(path_server)
        return False
    try:
        if os.path.exists(path_local) is False:
            os.makedirs(path_local)
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(copy_file)(filename, path_local) for filename in glob.glob(os.path.join(path_server, '*.*')))

        spatial_firing_path = path_server + '/MountainSort/DataFrames/spatial_firing.pkl'
        if os.path.isfile(spatial_firing_path) is True:
            if not os.path.isdir(path_local + '/MountainSort/DataFrames/'):
                os.makedirs(path_local + '/MountainSort/DataFrames/')
            shutil.copy(spatial_firing_path, path_local + '/MountainSort/DataFrames/spatial_firing.pkl')
        print('Copying is done, I will attempt to sort.')

    except Exception as ex:
        recording_to_sort = False
        add_to_list_of_failed_sortings(recording_to_sort)
        print('There is a problem with this file. '
              'I will move on to the next one. This is what Python says happened:')
        print(ex)
        return recording_to_sort
    return recording_to_sort


def get_next_recording_on_server_to_sort():
    recording_to_sort = False
    if not os.listdir(downtime_lists_path):
        return False
    else:
        list_to_read = os.listdir(downtime_lists_path)[0]
        list_to_read_path = downtime_lists_path + list_to_read
        if os.stat(list_to_read_path).st_size == 0:
            os.remove(list_to_read_path)
        else:
            downtime_file_reader = open(list_to_read_path, 'r+')
            recording_to_sort = downtime_file_reader.readlines()[0].strip()

            delete_processed_line(list_to_read_path)
            recording_to_sort = copy_recording_to_sort_to_local(recording_to_sort)
            if recording_to_sort is False:
                return False
            recording_to_sort = sorting_folder + recording_to_sort.split("/")[-1]

    return recording_to_sort


def monitor_to_sort():
    start_time = time.time()
    time_to_wait = 60.0
    while True:
        print('I am checking whether there is something to sort.')

        recording_to_sort = check_folder(sorting_folder)
        if recording_to_sort is not False:
            tags = get_tags_parameter_file(recording_to_sort)
            paired_recording = check_for_paired(tags)
            call_spike_sorting_analysis_scripts(recording_to_sort,
                                                tags,
                                                paired_recording=paired_recording)

        else:
            if os.environ.get('SINGLE_RUN'):
                print('Single run mode was active, so I will exit instead of monitoring the folders.')
                break

            print('Nothing urgent to sort. I will check if there is anything waiting on the server.')

            recording_to_sort = get_next_recording_on_server_to_sort()
            if recording_to_sort is not False:
                tags = get_tags_parameter_file(recording_to_sort)
                paired_recording = check_for_paired(tags)
                call_spike_sorting_analysis_scripts(recording_to_sort,
                                                    tags,
                                                    paired_recording=paired_recording)

            else:
                time.sleep(time_to_wait - ((time.time() - start_time) % time_to_wait))


def main():
    print('v - 0')
    print('-------------------------------------------------------------')
    print('This is a script that controls running the spike sorting analysis.')
    print('-------------------------------------------------------------')

    monitor_to_sort()


if __name__ == '__main__':
    main()