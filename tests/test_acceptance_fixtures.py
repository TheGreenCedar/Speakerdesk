"""Voice CLI contract with fabricated metadata, never acoustic/model proof."""
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import acceptance_fixtures as fixtures

IDENTIFIER='com.apple.voice.super-compact.fr-FR.Thomas'
LISTING=('Thomas (French (France)) fr_FR # Bonjour.\n'
         'Thomas (French (France)) fr_FR # Bonjour.\n'
         'Thomas (English (US)) en_US # Hello.\n')

class VoiceSelectorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        asset=self.root/'fr-FR/Thomas';asset.mkdir(parents=True)
        (asset/'Info.plist').write_bytes(plistlib.dumps({'MobileAssetProperties':{
            'VoiceId':IDENTIFIER,'Name':'Thomas','Languages':['fr-FR']}}))
        self.asset=asset
        self.patcher=patch.object(fixtures,'VOICE_ROOT',self.root);self.patcher.start()
    def tearDown(self):self.patcher.stop();self.temp.cleanup()
    def command(self,args,timeout):
        self.assertEqual(args,['/usr/bin/say','-v','?'])
        return subprocess.CompletedProcess(args,0,LISTING.encode(),'')
    def test_resolves_full_listed_name_and_locale_instead_of_resource_id(self):
        result=fixtures.voice(IDENTIFIER,self.command)
        self.assertEqual(result['say_selector'],'Thomas (French (France))')
        self.assertEqual(result['say_language'],'fr_FR')
        self.assertEqual(result['matching_listing_entries'],2)
        self.assertIn('footprint unverified',result['asset_binding'])
        self.assertIn('Info.plist',result['files_sha256'])
    def test_wrong_language_unknown_or_ambiguous_selector_fails_closed(self):
        for listing in ('Thomas en_US # Hello.\n','Other fr_FR # Bonjour.\n',
                        'Thomas fr_FR # Bonjour.\nThomas (French (France)) fr_FR # Bonjour.\n'):
            with self.subTest(listing=listing),self.assertRaisesRegex(ValueError,'unique listed CLI selector'):
                fixtures.voice(IDENTIFIER,lambda args,timeout:subprocess.CompletedProcess(args,0,listing,''))
    def test_absent_resource_never_uses_cli_fallback(self):
        with self.assertRaisesRegex(ValueError,'unavailable'):
            fixtures.voice('speakerdesk-no-such-voice',lambda *a,**kw:self.fail('Must not synthesize'))
    def test_generate_uses_resolved_selector_and_records_it(self):
        commands=[]
        def command(args,timeout):
            commands.append(args)
            if args==['/usr/bin/say','-v','?']:return self.command(args,timeout)
            if args[0]=='/usr/bin/afconvert':fixtures.write_wav(Path(args[-1]),[0,100,-100,0])
            else:Path(args[args.index('-o')+1]).write_bytes(b'fabricated AIFF')
            return subprocess.CompletedProcess(args,0,b'',b'')
        recipe={'id':'fabricated-selector-contract','parts':[{'kind':'speech','voice':IDENTIFIER,'text':'Bonjour'}]}
        destination=self.root/'output';fixtures.generate(recipe,destination,command)
        synthesis=next(args for args in commands if args[0]=='/usr/bin/say' and '-o' in args)
        self.assertEqual(synthesis[synthesis.index('-v')+1],'Thomas (French (France))')
        import json
        recorded=json.loads((destination/'synthesis.json').read_text())['speech'][0]['voice']
        self.assertEqual(recorded['say_selector'],'Thomas (French (France))')

    def test_successful_synthesis_with_empty_or_zero_audio_fails_closed(self):
        for index,samples in enumerate(([],[0]*160)):
            def command(args,timeout):
                if args==['/usr/bin/say','-v','?']:return self.command(args,timeout)
                if args[0]=='/usr/bin/afconvert':fixtures.write_wav(Path(args[-1]),samples)
                else:Path(args[args.index('-o')+1]).write_bytes(b'fabricated AIFF')
                return subprocess.CompletedProcess(args,0,b'',b'')
            recipe={'id':'empty-contract','parts':[{'kind':'speech','voice':IDENTIFIER,'text':'Bonjour'}]}
            destination=self.root/f'empty-{index}'
            with self.subTest(samples=len(samples)),self.assertRaisesRegex(ValueError,'empty or digital zero'):
                fixtures.generate(recipe,destination,command)
            self.assertFalse((destination/'replay.wav').exists())

if __name__=='__main__':unittest.main()
