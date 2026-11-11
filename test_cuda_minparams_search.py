"""CPU-only checks for scientific selection, architecture identity, and resumption."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cuda_minparams_search as search


class SearchTests(unittest.TestCase):
    def test_original_count_and_net_reduction_allow_local_growth(self):
        self.assertEqual(search.parameter_count(search.FRAME,search.baseline_spec()),27144960)
        frame,spec=search.standard(6,256,2048)
        search.validate(frame,spec)
        self.assertGreater(frame["d_ff"],1536)
        self.assertLess(search.parameter_count(frame,spec),27144960)
        frame,spec=search.standard(8,512,2048)
        with self.assertRaises(ValueError):search.validate(frame,spec)

    def test_hash_covers_frame_mode_sharing_order_and_entry_boundaries(self):
        frame,spec=search.standard(8,256,1024)
        identity=search.architecture_hash(frame,spec)
        renamed=copy.deepcopy(spec)
        renamed["name"]="cosmetic"
        self.assertEqual(identity,search.architecture_hash(frame,renamed))
        for field,value in (("d_model",320),("n_heads",8),("n_layers",9)):
            f=copy.deepcopy(frame);f[field]=value
            self.assertNotEqual(identity,search.architecture_hash(f,spec))
        for mutation in ("mode","share","order","split"):
            s=copy.deepcopy(spec)
            if mutation=="mode":s.pop("search_space")
            if mutation=="share":s["layers"][2]["attn"]=s["layers"][0]["attn"]
            if mutation=="order":s["layers"][0]["order"]="ffn_attn"
            if mutation=="split":
                s["layers"][0]={"attn":"A0"};s["layers"].insert(1,{"ffn":"F0"})
            self.assertNotEqual(identity,search.architecture_hash(frame,s))

    def test_breadth_diverse_and_expanded_ffn_order(self):
        valid=[]
        for name,frame,spec in search.breadth_candidates():
            try:search.validate(frame,spec)
            except ValueError:continue
            valid.append((name,frame,spec))
        self.assertGreater(len(valid),100)
        self.assertGreaterEqual(len({f["d_model"] for _,f,_ in valid}),6)
        self.assertTrue(any("ffa" in name for name,_,_ in valid))
        self.assertLess(min(search.parameter_count(f,s) for _,f,s in valid),10_000_000)

    def scheduler(self,root):
        args=search.make_parser().parse_args(["--root",str(root),"--deadline",search.iso(search.utc_now()+timedelta(hours=3)),
              "--training-cutoff",search.iso(search.utc_now()+timedelta(hours=2)),"--plan-only"])
        args.gpus=[0];args.runs_root=root.parent
        for name in ("specs","attempts","logs","inbox","inbox-processed"):(root/name).mkdir(parents=True,exist_ok=True)
        return search.Scheduler(args)

    def test_select_smallest_feasible_paired_three_seed_mean_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);scheduler=self.scheduler(root)
            baseline_id,_=scheduler.register("baseline",search.FRAME,search.baseline_spec(),baseline=True)
            candidates=[]
            for depth,loss in ((12,3.2),(8,3.39),(6,3.41)):
                frame,spec=search.standard(depth,256,1024)
                digest,_=scheduler.register(f"depth-{depth}",frame,spec)
                candidates.append(digest)
                for seed in (0,1,2):
                    job=scheduler.make_job(digest,seed,"confirmation")
                    summary={"best_dev_loss":loss,"parameters":search.parameter_count(frame,spec)}
                    scheduler.record(job,"completed",summary=summary)
                    scheduler.completed[job.run_id]={"job":job,"summary":summary,"path":str(root/"attempts"/job.run_id)}
            for seed,loss in enumerate((3.39,3.4,3.41)):
                job=scheduler.make_job(baseline_id,seed,"baseline")
                summary={"best_dev_loss":loss,"parameters":27144960}
                scheduler.record(job,"completed",summary=summary)
                scheduler.completed[job.run_id]={"job":job,"summary":summary,"path":str(root/"attempts"/job.run_id)}
            # A lucky one-seed tiny candidate must not qualify.
            frame,spec=search.standard(4,192,768);digest,_=scheduler.register("one-seed",frame,spec)
            job=scheduler.make_job(digest,0,"breadth");summary={"best_dev_loss":3.0}
            scheduler.record(job,"completed",summary=summary)
            scheduler.completed[job.run_id]={"job":job,"summary":summary,"path":str(root/"attempts"/job.run_id)}
            frozen=scheduler.freeze_selection()
            self.assertEqual(frozen["primary_architecture_id"],candidates[1])
            self.assertEqual(len(frozen["qualified_candidates"]),2)
            scheduler.stage="frozen";scheduler.close()
            # Torn final ledger line is repaired, preserving all complete attempts.
            with (root/"run_registry.jsonl").open("ab") as stream:stream.write(b'{"partial":')
            restored=self.scheduler(root)
            self.assertEqual(len(restored.completed),13)
            self.assertEqual(restored.freeze_selection()["primary_architecture_id"],candidates[1])
            restored.stage="frozen";restored.close()

    def test_malformed_and_oversize_architectures_are_logged(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);scheduler=self.scheduler(root)
            digest,_=scheduler.register("broken",search.FRAME,{"layers":None})
            self.assertIsNone(digest)
            frame,spec=search.standard(24,512,2048)
            self.assertIsNone(scheduler.register("huge",frame,spec)[0])
            records=[json.loads(line) for line in (root/"architecture_registry.jsonl").read_text().splitlines()]
            self.assertEqual([r["status"] for r in records],["rejected","rejected"])
            scheduler.stage="plan-only";scheduler.close()

    def test_duplicate_inbox_job_does_not_duplicate_active_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);scheduler=self.scheduler(root)
            frame,spec=search.standard(8,256,1024)
            digest,_=scheduler.register("queued",frame,spec)
            job=scheduler.make_job(digest,0,"breadth")
            scheduler.args.poll_seconds=.001
            with patch.object(scheduler,"ingest_inbox",return_value=[job]), patch.object(scheduler,"launch") as launch:
                scheduler.run_stage([job],"breadth",scheduler.cutoff)
                launch.assert_called_once_with(0,job)
            scheduler.stage="plan-only";scheduler.close()

    def test_inbox_proposal_registered_and_archived(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);scheduler=self.scheduler(root)
            frame,spec=search.standard(8,256,1024)
            (root/"inbox"/"proposal.json").write_text(json.dumps({"name":"proposal","frame":frame,"spec":spec}))
            jobs=scheduler.ingest_inbox()
            self.assertEqual(len(jobs),1)
            self.assertEqual(jobs[0].phase,"inbox")
            self.assertTrue((root/"inbox-processed"/"proposal.json").exists())
            scheduler.stage="plan-only";scheduler.close()


if __name__=="__main__":unittest.main()
