"""Time-bounded numerical child process. No hardware operations are allowed here."""
import sys
from . import workflow as w
from .session_models import fit_sample_records,optimize_one


def main():
    source,target=sys.argv[1:]
    request=w.read(source)
    payload=request["payload"]
    try:
        if request["operation"]=="fit":
            result=fit_sample_records(payload["config"],payload["manifest_path"])
        elif request["operation"]=="optimize":
            result=optimize_one(payload["config"],payload["final_v"],payload["charge_s"],payload.get("holds_policy"))
        else:
            raise ValueError("Unknown offline operation")
        w.save(target,dict(ok=True,result=result))
    except Exception as exc:
        w.save(target,dict(ok=False,error=f"{type(exc).__name__}: {exc}"))


if __name__=="__main__":
    main()

