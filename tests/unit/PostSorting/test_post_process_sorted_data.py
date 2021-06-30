import PostSorting.post_process_sorted_data
import numpy as np


def test_process_running_parameter_tag():
    tags = 'interleaveved_opto*test1*cat'
    result = PostSorting.post_process_sorted_data.process_running_parameter_tag(tags)
    desired_result = True, False
    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)

    tags = 'interleaved_opto*test1*cat*pixel_ratio=555'
    result = PostSorting.post_process_sorted_data.process_running_parameter_tag(tags)
    desired_result = True, 555
    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)


def main():
    test_process_running_parameter_tag()


if __name__ == '__main__':
    main()
