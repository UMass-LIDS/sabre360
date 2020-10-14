import unittest
import sabre360 as sbr



class MyTestCase(unittest.TestCase):
    #TiledBuffer tests

    def test_tiled_buffer(self):
        tb = sbr.TiledBuffer(4, 16)
        tb.put_in_buffer(0, 0, 4)
        tb.put_in_buffer(0, 4, 3)
        tb.put_in_buffer(1, 5, 2)
        tb.put_in_buffer(1, 3, 4)

        self.assertEqual(tb.segment_duration, 4)
        self.assertEqual(tb.tiles, 16)
        self.assertEqual(tb.get_buffer_element(0,0),4)
        self.assertEqual(tb.get_buffer_element(0, 4), 3)
        self.assertEqual(tb.get_buffer_element(1, 5), 2)
        self.assertEqual(tb.get_buffer_element(1, 3), 4)

        tb.play_out_buffer(3)

        self.assertEqual(tb.get_played_segment_partial(),3)
        self.assertEqual(tb.get_played_segments(),0)

        tb.play_out_buffer(2)

        self.assertEqual(tb.get_played_segment_partial(), 1)
        self.assertEqual(tb.get_played_segments(),1)





    #SessionInfo tests

    #HeadsetModel tests

    #UserModel tests

    #NetworkModel tests

    #Ewma tests

    #NavigationGraphPrediction

    #Session test



if __name__ == '__main__':
    unittest.main()
