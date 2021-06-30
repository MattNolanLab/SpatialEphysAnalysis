import control_sorting_analysis
import pytest


class TestGetSessionType:

    def test_openfield_type(self, tmp_path):
        parameters = '''openfield
        JohnWick/Open_field_opto_tagging_p038/M5_2018-03-06_15-34-44_of
        '''
        with open(tmp_path / 'parameters.txt', 'w') as f:
            f.write(parameters)

        session_type = control_sorting_analysis.get_session_type(str(tmp_path))

        assert session_type == 'openfield'        

    def test_vr_type(self, tmp_path):
        parameters = '''vr
        JohnWick/Open_field_opto_tagging_p038/M5_2018-03-06_15-34-44_of
        '''
        with open(tmp_path / 'parameters.txt', 'w') as f:
            f.write(parameters)

        session_type = control_sorting_analysis.get_session_type(str(tmp_path))

        assert session_type == 'vr'

    def test_invalid_type(self, tmp_path):
            parameters = '''openvr
            JohnWick/Open_field_opto_tagging_p038/M5_2018-03-06_15-34-44_of
            '''
            with open(tmp_path / 'parameters.txt', 'w') as f:
                f.write(parameters)

            session_type = control_sorting_analysis.get_session_type(str(tmp_path))

            assert session_type == 'openvr'
            
    def test_file_is_dir(self, tmp_path):
        with pytest.raises(Exception):
            is_vr, is_open_field = control_sorting_analysis.get_session_type(str(tmp_path))
