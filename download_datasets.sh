#!/bin/bash
# Interactive dataset downloader for RoboCasa/OpenPI
# Usage: bash download_datasets.sh

set -e

ATOMIC_SEEN=(
    CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug
    NavigateKitchen OpenCabinet OpenDrawer OpenStandMixerHead
    PickPlaceCounterToCabinet PickPlaceCounterToStove PickPlaceDrawerToCounter
    PickPlaceSinkToCounter PickPlaceToasterToCounter SlideDishwasherRack
    TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet
)

COMPOSITE_SEEN=(
    DeliverStraw GetToastedBread KettleBoiling LoadDishwasher
    PackIdenticalLunches PreSoakPan PrepareCoffee RinseSinkBasin
    ScrubCuttingBoard SearingMeat SetUpCuttingStation StackBowlsCabinet
    SteamInMicrowave StirVegetables StoreLeftoversInBowl WashLettuce
)

COMPOSITE_UNSEEN=(
    ArrangeBreadBasket ArrangeTea BreadSelection CategorizeCondiments
    CuttingToolSelection GarnishPancake GatherTableware HeatKebabSandwich
    MakeIceLemonade PanTransfer PortionHotDogs RecycleBottlesByType
    SeparateFreezerRack WaffleReheat WashFruitColander WeighIngredients
)

ALL_TASKS=("${ATOMIC_SEEN[@]}" "${COMPOSITE_SEEN[@]}" "${COMPOSITE_UNSEEN[@]}")

echo "=== RoboCasa Dataset Downloader ==="
echo ""
echo "Available task groups:"
echo "  [1] atomic_seen      (18 tasks, ~12 GB)"
echo "  [2] composite_seen   (16 tasks)"
echo "  [3] composite_unseen (16 tasks)"
echo "  [4] all              (50 tasks)"
echo ""
echo "Or enter task names separated by commas, e.g.: CloseBlenderLid,OpenCabinet"
echo ""
echo "Available tasks:"
for i in "${!ALL_TASKS[@]}"; do
    printf "  %-40s" "${ALL_TASKS[$i]}"
    if (( (i + 1) % 3 == 0 )); then echo ""; fi
done
echo ""
echo ""
read -p "Selection: " selection

# Parse selection
TASKS=()
case "$selection" in
    1|atomic_seen)
        TASKS=("${ATOMIC_SEEN[@]}")
        ;;
    2|composite_seen)
        TASKS=("${COMPOSITE_SEEN[@]}")
        ;;
    3|composite_unseen)
        TASKS=("${COMPOSITE_UNSEEN[@]}")
        ;;
    4|all)
        TASKS=("${ALL_TASKS[@]}")
        ;;
    *)
        IFS=',' read -ra TASKS <<< "$selection"
        # Trim whitespace
        for i in "${!TASKS[@]}"; do
            TASKS[$i]=$(echo "${TASKS[$i]}" | xargs)
        done
        ;;
esac

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "No tasks selected. Exiting."
    exit 1
fi

echo ""
echo "Downloading ${#TASKS[@]} task(s): ${TASKS[*]}"
echo ""

# Build Python list string
TASK_LIST=$(printf "'%s'," "${TASKS[@]}")
TASK_LIST="[${TASK_LIST%,}]"

python -c "
from robocasa.scripts.download_datasets import download_datasets
download_datasets(split=['target'], tasks=${TASK_LIST}, source=['human'])
"

echo ""
echo "Done. Datasets saved under <robocasa clone>/datasets/v1.0/ (default ~/robocasa/datasets/v1.0/; override via DATASET_BASE_PATH in robocasa/macros_private.py)."
