"""CPU conservation/bounds tests; fabricated cells are not acoustic accuracy."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from context_plan import requests,CORE,CONTEXT,PHYSICAL_MAX,POLICY
from test_canonical_assembly import book_for,part
import test_canonical_runtime as runtime_fixtures


def repeated(book,row):
    req=book.decode_requests(row['id'],contextual=True)
    return [part(req[0],'head go go 12.5 tail',[(1000,5000),(180000,181000),
                (190000,193000),(202000,203000),(239680,240000)]),
            part(req[1],'go go 12.5 after',[(180000,181000),(190000,193000),
                (202000,203000),(300000,310000)])]


class ContextPolicyTests(unittest.TestCase):
    def test_balanced_original_audio_requests_are_revision_bound_and_physically_bounded(self):
        for total in (1,24*16000+1,30*16000,357*16000+829,2*3600*16000):
            row={'id':'one','start_sample':32000,'end_sample':32000+total,
                 'machine_revision':4,'audio_revision':9,'language_epoch':2}
            req=requests(row)
            self.assertEqual((req[0]['core_start_sample'],req[-1]['core_end_sample']),
                             (row['start_sample'],row['end_sample']))
            self.assertEqual(sum(p['core_end_sample']-p['core_start_sample'] for p in req),total)
            self.assertTrue(all(x['core_end_sample']==y['core_start_sample'] for x,y in zip(req,req[1:])))
            for p in req:
                self.assertLessEqual(p['core_end_sample']-p['core_start_sample'],CORE)
                self.assertLessEqual(p['end_sample']-p['start_sample'],PHYSICAL_MAX)
                self.assertEqual(p['start_sample'],max(row['start_sample'],p['core_start_sample']-CONTEXT))
                self.assertEqual(p['end_sample'],min(row['end_sample'],p['core_end_sample']+CONTEXT))
                self.assertEqual((p['machine_revision'],p['audio_revision'],p['language_epoch']),(4,9,2))
        for edges in ([0,14*16000+1,28*16000],[0,100,99,28*16000],[0,True,28*16000]):
            with self.assertRaises(ValueError):requests(dict(row,start_sample=0,end_sample=28*16000),edges=edges)

    def test_independent_repeats_survive_and_corroborated_seam_unit_appears_once(self):
        book,row=book_for(seconds=24)
        current=book.apply_decode_parts(row['id'],repeated(book,row),stage='refined',contextual=True)
        self.assertEqual(current['text'],'head go go 12.5 after')
        self.assertEqual(current['text'].split().count('go'),2)
        self.assertFalse(current['assembly_provenance']['alignment_complete'])
        self.assertEqual(current['assembly_provenance']['context_policy'],POLICY)
        near=[w for w in current['assembly_provenance']['words'] if w['core_ownership']=='unknown']
        self.assertEqual([w['text'] for w in near],['go'])
        self.assertIsNone(near[0]['start_sample']);self.assertNotIn('alignment',current)

    def test_mismatch_or_protected_correction_preserves_primary_and_all_raw_candidates(self):
        for protected,mismatch in [(False,True),(True,False)]:
            book,row=book_for(seconds=24)
            row=book.apply_model(row['id'],0,'Prior words retained',start_sample=0,end_sample=24*16000,
                                 stage='live',complete=True)
            if protected:row=book.edit(row['id'],row['machine_revision'],'Human correction retained')
            before=copy.deepcopy(row);parts=repeated(book,row)
            if mismatch:parts[1]['alignment']['words'][1]['start_sample']=195000
            current=book.apply_decode_parts(row['id'],parts,stage='refined',contextual=True)
            self.assertEqual(current['text'],before['text']);self.assertEqual(current['machine_revision'],before['machine_revision'])
            self.assertNotIn('alignment',current)
            self.assertEqual(current['machine_versions'][-1]['parts'],parts)
            self.assertEqual(current['machine_versions'][-1]['complete'],not mismatch)

    def test_failed_english_context_has_explicit_review_without_disjoint_success_fallback(self):
        for mode in ('en','auto'):
            case=runtime_fixtures.CanonicalTests();case.setUp();case.engine.config['authoritative_tail']=False
            try:
                case.peer.alignment_supported=lambda language:language=='en'
                case.peer.align_canonical=lambda *args,**kwargs:None
                case.engine.timeline[0]['language']=mode
                case.feed(0,30,mode=mode);case.engine.handle({'type':'stop'})
                row=case.rows()[-1]
                self.assertEqual(row['text'],case.peer.text)
                self.assertEqual(row['canonical_unresolved'],'unresolved_alignment')
                self.assertEqual(row['transcription_review']['reason'],'canonical_ownership_unresolved')
                events=[json.loads(l) for l in case.engine.canonical.archive.path.read_text().splitlines()]
                retained=next(e for e in events if e['type']=='bounded_machine_version')
                self.assertEqual(retained['version']['context_policy'],POLICY)
                self.assertEqual(len(retained['parts']),3)
                self.assertTrue(all(p['text']==case.peer.text for p in retained['parts']))
                self.assertFalse(any(e['type']=='disjoint_core_machine_version' for e in events))
                self.assertLessEqual(max(case.peer.calls[-3:]),PHYSICAL_MAX)
            finally:case.tearDown()

    def test_failed_auto_context_language_cannot_enable_disjoint_retry(self):
        case=runtime_fixtures.CanonicalTests();case.setUp();case.engine.config['authoritative_tail']=False
        try:
            case.peer.alignment_supported=lambda language:language=='en'
            case.peer.align_canonical=lambda *args,**kwargs:None
            case.engine.timeline[0]['language']='auto'
            case.feed(0,30,mode='auto')
            row=next(iter(case.engine.canonical.book.rows.values()))
            case.engine.canonical.decode(row,'live')
            original=copy.deepcopy(row);calls=case.peer.transcribe
            def french(audio,language,names,overlap=False):
                parts=calls(audio,language,names,overlap)
                for part in parts:part['language']='fr'
                return parts
            case.peer.transcribe=french
            for _ in range(2):
                case.engine.canonical.decode(row,'live')
                self.assertEqual(row['language'],'en')
                self.assertEqual(row['machine_revision'],original['machine_revision'])
                self.assertEqual(row['text'],original['text'])
                self.assertEqual(row['canonical_unresolved'],'unresolved_alignment')
                self.assertEqual(row['context_candidate_language']['language'],'fr')
            events=[json.loads(l) for l in case.engine.canonical.archive.path.read_text().splitlines()]
            self.assertFalse(any(e['type']=='disjoint_core_machine_version' for e in events))
            # An explicit user language epoch starts a new independent route.
            case.engine.handle({'type':'language','generation':1,'language':'fr','start_sample':30*16000})
            case.feed(30,60,'fr',1);case.engine.handle({'type':'stop'})
            latest=case.engine.canonical.book.rows[case.engine.canonical.book.order[-1]]
            self.assertEqual(latest['language_epoch'],1);self.assertEqual(latest['language'],'fr')
            self.assertNotIn('canonical_unresolved',latest)
            self.assertEqual(latest['bounded_decode_provenance']['method'],'disjoint_original_audio_cores')
        finally:case.tearDown()


if __name__=='__main__':unittest.main()
