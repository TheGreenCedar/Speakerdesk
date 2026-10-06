"""Lossless bounded acoustic evidence storage; fabricated metadata, no audio."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from speech_admission import FrameArchive,SpeechFrames,INPUT_POLICY

class SpeechArchiveTests(unittest.TestCase):
    def test_lossless_historical_queries_final_blocking_and_bounded_index(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(FrameArchive,'MAX_INDEX',2):
            archive=FrameArchive(Path(folder)/'frames');frames=SpeechFrames(max_frames=31,archive=archive)
            expected=[]
            for n in range(800):
                observation={'input_policy':INPUT_POLICY,'raw_probability':(n%71)/71,
                    'normalized_probability':(n%89)/89,'gain':1.+n%255,
                    'constant_value':None if n%3 else 0.}
                frames.append(n*512,(n+1)*512,(n%97)/97,observation=observation)
                expected.append(json.loads(json.dumps(frames.frames[-1])))
            frames.persist()
            self.assertEqual(archive.query(0,800*512),expected)
            self.assertEqual(archive.query(211*512+1,214*512-1),expected[211:214])
            self.assertEqual(archive.query(798*512,800*512),expected[-2:])
            self.assertLessEqual(len(archive.index),2)
            self.assertLess(archive.path.stat().st_size,len(json.dumps(expected).encode()))
            with archive.path.open('rb') as source:
                while header:=source.read(archive.HEADER.size):
                    magic,size,raw,a,b,digest=archive.HEADER.unpack(header)
                    self.assertEqual(magic,b'SVF1');self.assertLessEqual(raw,archive.MAX_RAW)
                    self.assertLessEqual((b-a)//512,archive.MAX_FRAMES);source.seek(size,1)
    def test_corrupt_payload_header_truncation_and_schema_fail_closed(self):
        for damage in ('payload','header','truncate','schema'):
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as folder:
                archive=FrameArchive(Path(folder)/'frames');archive.append([(0,512,.01,False)])
                original=archive.path.read_bytes();bad=bytearray(original)
                if damage=='payload':bad[-1]^=1
                elif damage=='schema':bad[0]=ord('X')
                elif damage=='truncate':bad=bad[:-1]
                else:struct.pack_into('<I',bad,8,archive.MAX_RAW+1)
                archive.path.write_bytes(bad)
                with self.assertRaises((ValueError,zlib.error)):
                    archive.query(0,512)
    def test_append_failure_does_not_advance_index_or_remove_original_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            archive=FrameArchive(Path(folder)/'frames');archive.append([(0,512,.01,False)])
            original=archive.path.read_bytes();index=archive.index.copy()
            with patch('speech_admission.zlib.compress',side_effect=ValueError('Fixture compression fault')):
                with self.assertRaises(ValueError):archive.append([(512,1024,.02,False)])
            self.assertEqual(archive.path.read_bytes(),original);self.assertEqual(archive.index,index)
            self.assertEqual(archive.query(0,512),[[0,512,.01,False]])
