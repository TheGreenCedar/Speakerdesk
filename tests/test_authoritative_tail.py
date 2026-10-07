"""Fabricated CPU cells/state only; never acoustic or ASR accuracy proof."""
import copy,json,re,sys,tempfile,unittest,wave
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from authoritative_tail import assemble,requests,validated_cells,POLICY
from authority_store import AuthorityStore
from word_alignment import prepare_text,materialize,TokenAnchor,FrameClock,AcceptancePolicy
from reading_turns import CALIBRATION
from test_canonical_assembly import book_for
import test_canonical_runtime as runtime_fixtures
VOCAB=('<s>','<pad>','</s>','<unk>',' ',*'abcdefghijklmnopqrstuvwxyz0123456789')


def part(request,text,times):
    prepared=prepare_text(text,VOCAB);units=list(re.finditer(r'\S+',text));anchors=[];last=-2;word_frames={}
    for unit,start in zip(units,times):word_frames[unit.start()]=round((round(start*16000)-request['start_sample'])/320)
    for i,t in enumerate(prepared.targets):
        unit=next((w for w in units if w.start()<=t.start_char<w.end()),None)
        frame=max(last+2,word_frames[unit.start()]) if unit else last+2
        if unit:word_frames[unit.start()]=frame+2
        anchors.append(TokenAnchor(i,t.token_id,frame,-1,last+1,-1));last=frame
    result=materialize(prepared,anchors,clock=FrameClock(320,0,CALIBRATION),policy=AcceptancePolicy(-20,CALIBRATION),
                       audio_start_sample=request['start_sample'],audio_num_samples=request['end_sample']-request['start_sample'])
    assert result['complete']
    return {'request':copy.deepcopy(request),'text':text,'complete':True,'alignment':result,'passages':[]}


class AuthoritativeTailTests(unittest.TestCase):
    def pair(self,first='Head  go tail',second='Different Go after',times1=(2,13.8,15.5),times2=(11.9,13.8,16)):
        book,row=book_for(seconds=28);req=requests(row)
        return book,row,req,[part(req[0],first,times1),part(req[1],second,times2)]
    def test_other_context_wording_does_not_veto_authoritative_prefix_tail(self):
        book,row,req,parts=self.pair();result=assemble(row,req,parts,vocabulary=VOCAB)
        self.assertTrue(result['complete']);self.assertEqual(result['text'],'Head  go after')
        self.assertTrue(result['rollover_receipts'][0]['bidirectional_unique'])
        self.assertGreater(result['rollover_receipts'][0]['intersecting_cell_count'],0)
        applied=book.apply_authoritative_parts(row['id'],parts,stage='refined',vocabulary=VOCAB)
        self.assertEqual(applied['text'],result['text']);self.assertEqual(applied['assembly_provenance']['context_policy'],POLICY)
    def test_whole_authoritative_revision_replaces_provisional_text_without_timing(self):
        book,row=book_for(seconds=7);req=requests(row)
        book.apply_model(row['id'],0,'Older provisional words',start_sample=0,end_sample=112000,stage='live',complete=True)
        req=requests(book.rows[row['id']]);chosen={'request':req[0],'text':'  New café 12.5 go go 👩🏽‍💻  ','complete':True,'alignment':None}
        current=book.apply_authoritative_parts(row['id'],[chosen],stage='refined')
        self.assertEqual(current['text'],chosen['text']);self.assertTrue(current['recognition_complete'])
        self.assertTrue(all(w['start_sample'] is None for w in current['assembly_provenance']['words']))
        self.assertEqual(current['machine_versions'][0]['text'],'Older provisional words')
    def test_rapid_distinct_events_cannot_map_by_proximity_alone(self):
        book,row,req,parts=self.pair(second='Different go after',times2=(11.9,14.0,16))
        result=assemble(row,req,parts,vocabulary=VOCAB)
        self.assertFalse(result['complete']);self.assertTrue(result['recognition_complete'])
        self.assertEqual(result['reason'],'boundary_mapping_absent');self.assertEqual(result['local_seams'][0]['state'],'finite_review')
    def test_one_new_word_cannot_consume_two_old_repeat_events(self):
        book,row,req,parts=self.pair(first='head go go tail',second='head go after',times1=(2,13.8,14.0,15.5),times2=(11.9,13.8,16))
        # A fabricated complete new path spans the g of the first event and o
        # of the second. Forward uniqueness alone loses one genuine repeat.
        aligned=parts[1]['alignment'];w=next(w for w in aligned['words'] if w['text']=='go')
        c=next(c for c in aligned['characters'] if c['start_char']==w['start_char']+1)
        c.update(frame=152,start_sample=224640,end_sample=224960)
        # Right physical anchor is 11s: 11s+152*20ms =14.04s.
        c['start_sample']=req[1]['start_sample']+152*320;c['end_sample']=c['start_sample']+320
        w['end_sample']=c['end_sample']
        space=next(c for c in aligned['characters'] if c['start_char']==w['end_char'])
        space.update(frame=153,start_sample=req[1]['start_sample']+153*320,end_sample=req[1]['start_sample']+154*320)
        result=assemble(row,req,parts,vocabulary=VOCAB)
        self.assertFalse(result['complete']);self.assertEqual(result['reason'],'ambiguous_bidirectional_boundary')

    def test_genuine_repetitions_at_distinct_cells_survive(self):
        book,row,req,parts=self.pair(first='head go go tail',second='head go go after',times1=(2,13.8,14.0,15.5),times2=(11.9,13.8,14.0,16))
        result=assemble(row,req,parts,vocabulary=VOCAB)
        self.assertTrue(result['complete']);self.assertEqual(result['text'],'head go go after')
    def test_truncated_extra_raw_letters_and_reordered_cells_cannot_be_consumed(self):
        for fault in ('truncate','reverse','duplicate','weak'):
            book,row,req,parts=self.pair(second='Different goat after')
            chars=parts[1]['alignment']['characters'];word=next(w for w in parts[1]['alignment']['words'] if w['text']=='goat')
            if fault=='truncate':parts[1]['alignment']['characters']=[c for c in chars if not word['start_char']+2<=c['start_char']<word['end_char']]
            elif fault=='reverse':parts[1]['alignment']['characters']=list(reversed(chars))
            elif fault=='duplicate':chars.insert(1,copy.deepcopy(chars[0]))
            else:chars[0]['log_probability']=-21
            result=assemble(row,req,parts,vocabulary=VOCAB)
            self.assertFalse(result['complete']);self.assertIsNone(result['text'])
    def test_raw_bounds_and_cas_protect_primary_human_text(self):
        book,row,req,parts=self.pair()
        forged=copy.deepcopy(req);forged[0]['start_sample']=-1;altered=copy.deepcopy(parts);altered[0]['request']=forged[0]
        with self.assertRaises(ValueError):assemble(row,forged,altered,vocabulary=VOCAB)
        book.edit(row['id'],row['machine_revision'],'Human correction retained')
        with self.assertRaises(ValueError):book.apply_authoritative_parts(row['id'],parts,stage='refined',vocabulary=VOCAB)
        fresh=requests(book.rows[row['id']]);parts=[part(fresh[0],'Head go tail',(2,13.8,15.5)),part(fresh[1],'Different Go after',(11.9,13.8,16))]
        current=book.apply_authoritative_parts(row['id'],parts,stage='refined',vocabulary=VOCAB)
        self.assertEqual(current['text'],'Human correction retained');self.assertNotIn('reading_word_evidence',current)
    def test_overlapping_cross_context_envelope_becomes_null_without_text_loss(self):
        book,row,req,parts=self.pair(second='a go after',times1=(2,12,15.5),times2=(11.9,12,12.36))
        for index,o_time in ((0,12.4),(1,12.32)):
            aligned=parts[index]['alignment'];word=next(w for w in aligned['words'] if w['text'].casefold()=='go')
            for start_char,seconds in ((word['start_char']+1,o_time),(word['end_char'],o_time+.02)):
                cell=next(c for c in aligned['characters'] if c['start_char']==start_char)
                frame=round((seconds*16000-req[index]['start_sample'])/320)
                cell.update(frame=frame,start_sample=req[index]['start_sample']+frame*320,end_sample=req[index]['start_sample']+(frame+1)*320)
            word['end_sample']=round((o_time+.02)*16000)
        result=assemble(row,req,parts,vocabulary=VOCAB)
        self.assertTrue(result['complete']);self.assertEqual(result['text'],'Head  go after')
        self.assertFalse(result['alignment_complete']);self.assertIsNone(result['words'][-1]['start_sample'])

    def test_current_cas_rebinding_keeps_original_raw_decode_id(self):
        book,row,req,parts=self.pair()
        for p in parts:p['source_inference_request']=copy.deepcopy(p['request'])
        original=assemble(row,req,parts,vocabulary=VOCAB)
        row['machine_revision']+=1;row['audio_revision']+=1
        fresh=requests(row)
        for p,r in zip(parts,fresh):p['request']=r
        rebound=assemble(row,fresh,parts,vocabulary=VOCAB)
        self.assertTrue(rebound['complete']);self.assertEqual([w['decode_id'] for w in rebound['words']],[w['decode_id'] for w in original['words']])
        self.assertNotEqual(fresh[0]['machine_revision'],parts[0]['source_inference_request']['machine_revision'])

    def test_mature_rollover_cannot_rewind_and_provisional_boundary_can_change(self):
        book,row=book_for(seconds=42);row['state']='open';req=requests(row)
        parts=[part(req[0],'Alpha go go tail',(12,13.8,14,15.5)),
            part(req[1],'Alpha go go midway end tail',(12,13.8,14,24,27.8,29.5)),
            part(req[2],'Earlier end after',(25.5,27.8,31))]
        for p in parts:p['source_inference_request']=copy.deepcopy(p['request'])
        result=assemble(row,req,parts,vocabulary=VOCAB)
        self.assertTrue(result['complete']);self.assertTrue(result['rollover_receipts'][0]['committed'])
        self.assertFalse(result['rollover_receipts'][1]['committed'])
        locked={r['nominal_frontier_sample']:r for r in result['rollover_receipts'] if r['committed']}
        changed=copy.deepcopy(parts);word=[w for w in changed[1]['alignment']['words'] if w['text']=='go'][1]
        for cell in changed[1]['alignment']['characters']:
            if word['start_char']<=cell['start_char']<=word['end_char']:
                cell['frame']+=20;cell['start_sample']+=6400;cell['end_sample']+=6400
        word['start_sample']+=6400;word['end_sample']+=6400
        provisional=assemble(row,req,changed,vocabulary=VOCAB)
        self.assertTrue(provisional['complete'])
        self.assertLess(provisional['rollover_receipts'][0]['left_raw_end_char'],result['rollover_receipts'][0]['left_raw_end_char'])
        rejected=assemble(row,req,changed,vocabulary=VOCAB,committed_receipts=locked)
        self.assertFalse(rejected['complete']);self.assertEqual(rejected['reason'],'committed_authority_changed')
        # Source identity remains fixed while current publication CAS changes.
        row['machine_revision']+=1;row['audio_revision']+=1;fresh=requests(row)
        for p,r in zip(parts,fresh):p['request']=r
        rebound=assemble(row,fresh,parts,vocabulary=VOCAB,committed_receipts=locked)
        self.assertTrue(rebound['complete'])
        parts[0]['source_audio_float32_sha256']='forged-original-PCM'
        self.assertEqual(assemble(row,fresh,parts,vocabulary=VOCAB,committed_receipts=locked)['reason'],'committed_authority_changed')

    def test_earlier_supported_anchor_preserves_new_raw_tail_and_rejection_receipt(self):
        book,row,req,parts=self.pair(first='Alpha campaign tail',second='Alpha campaign after',times1=(12,13.8,15.5),times2=(12,13.8,16))
        word=next(w for w in parts[1]['alignment']['words'] if w['text']=='campaign')
        for cell in parts[1]['alignment']['characters']:
            if word['start_char']<=cell['start_char']<=word['end_char']:
                cell['frame']+=20;cell['start_sample']+=6400;cell['end_sample']+=6400
        word['start_sample']+=6400;word['end_sample']+=6400
        result=assemble(row,req,parts,vocabulary=VOCAB)
        self.assertTrue(result['complete']);self.assertEqual(result['text'],'Alpha campaign after')
        receipt=result['rollover_receipts'][0]
        self.assertEqual(receipt['left_boundary_word']['text'],'Alpha')
        self.assertEqual(receipt['later_unmapped_raw_anchors'][0]['text'],'campaign')

    def test_unchanged_or_between_cadence_authoritative_state_does_not_invent_failure(self):
        import numpy as np
        for enabled,reason,grow in ((True,None,False),(True,'ambiguous_bidirectional_boundary',False),(False,None,False),(True,None,True)):
            case=runtime_fixtures.CanonicalTests();case.setUp()
            try:
                book,row=book_for(seconds=30);row=book.rows[row['id']];row['state']='open';row['text']='Accepted raw words'
                case.engine.canonical.book=book
                case.peer.align_canonical=lambda *a,**kw:None;case.peer.alignment_supported=lambda language:language=='en'
                case.engine.config['authoritative_tail']=enabled
                case.peer.feed(np.zeros(30*16000,dtype=np.float32),final=True)
                case.engine.received=30*16000;case.engine.processed=30
                case.engine.canonical.last_decoded[row['id']]=(row['end_sample'],row['audio_revision'],'live')
                if reason:row['canonical_unresolved']=reason
                if grow:
                    case.peer.feed(np.zeros(16000,dtype=np.float32),final=True)
                    book._set_end(row,31*16000);book.cursor=31*16000;case.engine.received=31*16000;case.engine.processed=31
                case.engine.canonical.commit()
                expected=reason if enabled else 'unresolved_alignment'
                self.assertEqual(row.get('canonical_unresolved'),expected)
                self.assertEqual(case.peer.calls,[])
            finally:case.tearDown()

    def test_hot_eviction_preserves_all_33_selected_authorities(self):
        with tempfile.TemporaryDirectory() as folder:
            store=AuthorityStore(Path(folder)/'store',max_entries=32,max_bytes=1024)
            for i in range(33):store.put(('epoch',i),{'text':f'Raw authority {i}'})
            for i in range(33):self.assertEqual(store.get(('epoch',i))['text'],f'Raw authority {i}')
            self.assertLessEqual(len(store.hot),32);self.assertLessEqual(store.bytes,1024)
            self.assertIsNone(store.get(('new_operation',0)))
    def test_actual_engine_33_core_second_sweep_makes_zero_asr_calls(self):
        case=runtime_fixtures.CanonicalTests();case.setUp()
        try:
            with wave.open(str(case.path),'wb') as w:w.setparams((1,2,16000,0,'NONE','none'));w.writeframes(b'\x01\x00'*462*16000)
            book,row=book_for(seconds=462);case.engine.canonical.book=book
            case.peer.align_canonical=lambda *a,**kw:None;case.peer.alignment_supported=lambda language:language=='en'
            case.engine.canonical.decode(row,'live');calls=len(case.peer.calls)
            self.assertEqual(calls,33)
            case.engine.canonical.decode(row,'live');self.assertEqual(len(case.peer.calls),calls)
            self.assertEqual(len(list(case.engine.canonical.authoritative_store.directory.glob('*.json'))),33)
        finally:case.tearDown()
    def test_cancelled_operation_and_new_retry_never_reuse_old_asr_authority(self):
        case=runtime_fixtures.CanonicalTests();case.setUp()
        try:
            book,row=book_for(seconds=30);case.engine.canonical.book=book;req=requests(row)
            case.peer.align_canonical=lambda *a,**kw:None
            case.engine.canonical.refinement_request={'operation_id':'cancelled','language_epoch':0}
            original=case.peer.transcribe
            def cancel(*args,**kw):
                result=original(*args,**kw);case.inbox.push({'type':'cancel_refinement','operation_id':'cancelled'});return result
            case.peer.transcribe=cancel
            self.assertFalse(case.engine.canonical.decode_long_parts(row,'refined',req,aligned=True,authoritative=True))
            self.assertEqual(row['text'],'');before=len(case.peer.calls)
            case.peer.transcribe=original;case.engine.canonical.refinement_request={'operation_id':'new-retry','language_epoch':0}
            parts=case.engine.canonical.decode_long_parts(row,'refined',req,aligned=True,authoritative=True)
            self.assertEqual(len(parts),len(req));self.assertEqual(len(case.peer.calls)-before,len(req))
            self.assertEqual(row['text'],'')
        finally:case.tearDown()


if __name__=='__main__':unittest.main()
