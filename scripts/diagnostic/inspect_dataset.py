## { SCRIPT

##
## === DEPENDENCIES
##

from pathlib import Path
from ww_quokka_sims.sim_io import load_dataset
from jormi.ww_io import log_manager
import utils

##
## === OPERATOR CLASS
##


class ScriptInterface:

    def __init__(
        self,
        *,
        dataset_dir: Path,
        dataset_tag: str,
    ):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.dataset_tag = dataset_tag
        self._validate_inputs()

    def _validate_inputs(
        self,
    ) -> None:
        utils.ensure_looks_like_boxlib_dir(
            dataset_dir=self.dataset_dir,
        )

    def run(
        self,
    ) -> None:
        with load_dataset.QuokkaDataset(dataset_dir=self.dataset_dir, verbose=True) as ds:
            ds.list_available_field_keys()


##
## === PROGRAM MAIN
##


def main():
    user_args = utils.get_user_args()
    script_interface = ScriptInterface(
        dataset_dir=user_args.dir,
        dataset_tag=user_args.tag,
        ## TODO: add field to check diagnostics for
    )
    script_interface.run()


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
