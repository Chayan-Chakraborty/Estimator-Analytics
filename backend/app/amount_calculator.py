import json
import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class AmountCalculatorUtils:
    INCH_TO_FEET = 0.00694444
    MM_TO_FEET = 0.0000107639111056
    INCHES = "inches"
    MM = "mm"
    PER_SQ_FT = "Per sq-ft"
    SELECTED_OPTION = "selectedOption"
    LENGTH = "length"
    WIDTH = "width"
    VALUE = "value"

    QUA_LF_ATTR = "QUA_LF_ATTR"
    PPU_LF_ATTR = "PPU_LF_ATTR"
    QNT_ACCS_ATTR = "QNT_ACCS_ATTR"
    PPU_ACCS_ATTR = "PPU_ACCS_ATTR"
    RAT_OTH_ATTR = "RAT_OTH_ATTR"
    MES_OTH_ATTR = "MES_OTH_ATTR"
    RAT_FC_ATTR = "RAT_FC_ATTR"
    MES_FC_ATTR = "MES_FC_ATTR"
    RAT_WD_ATTR = "RAT_WD_ATTR"
    MES_WD_ATTR = "MES_WD_ATTR"

    @staticmethod
    def calc_item_amount(type_identifier: str, item) -> float:
        if item is None:
            return 0.0

        if type_identifier == "WD":
            return AmountCalculatorUtils.calc_woodwork_amount(item)
        elif type_identifier == "FC":
            return AmountCalculatorUtils.calc_false_ceiling_amount(item)
        elif type_identifier == "ACS":
            return AmountCalculatorUtils.calc_accessories_amount(item)
        elif type_identifier == "LF":
            return AmountCalculatorUtils.calc_loose_furniture_amount(item)
        elif type_identifier == "OTH":
            return AmountCalculatorUtils.calc_other_service_amount(item)
        return 0.0

    @staticmethod
    def calc_woodwork_amount(item) -> float:
        m = AmountCalculatorUtils.convert_room_attribute_json(item.attributes)
        return AmountCalculatorUtils.calc_measurement_form(m, AmountCalculatorUtils.MES_WD_ATTR, AmountCalculatorUtils.RAT_WD_ATTR) if AmountCalculatorUtils.RAT_WD_ATTR in m else 0.0

    @staticmethod
    def calc_false_ceiling_amount(item) -> float:
        m = AmountCalculatorUtils.convert_room_attribute_json(item.attributes)
        return AmountCalculatorUtils.calc_measurement_form(m, AmountCalculatorUtils.MES_FC_ATTR, AmountCalculatorUtils.RAT_FC_ATTR) if AmountCalculatorUtils.RAT_FC_ATTR in m else 0.0

    @staticmethod
    def calc_other_service_amount(item) -> float:
        m = AmountCalculatorUtils.convert_room_attribute_json(item.attributes)
        return AmountCalculatorUtils.calc_measurement_form(m, AmountCalculatorUtils.MES_OTH_ATTR, AmountCalculatorUtils.RAT_OTH_ATTR) if AmountCalculatorUtils.RAT_OTH_ATTR in m else 0.0

    @staticmethod
    def calc_accessories_amount(item) -> float:
        m = AmountCalculatorUtils.convert_room_attribute_json(item.attributes)
        qty_str = m.get(AmountCalculatorUtils.QNT_ACCS_ATTR, {}).get(AmountCalculatorUtils.VALUE, "0")
        price_str = m.get(AmountCalculatorUtils.PPU_ACCS_ATTR, {}).get(AmountCalculatorUtils.VALUE, "0")
        qty = AmountCalculatorUtils.parse_double_or_default(qty_str, 0.0)
        price = AmountCalculatorUtils.parse_double_or_default(price_str, 0.0)
        return AmountCalculatorUtils.round_double(qty * price, 2)

    @staticmethod
    def calc_loose_furniture_amount(item) -> float:
        m = AmountCalculatorUtils.convert_room_attribute_json(item.attributes)
        qty_str = m.get(AmountCalculatorUtils.QUA_LF_ATTR, {}).get(AmountCalculatorUtils.VALUE, "0")
        price_str = m.get(AmountCalculatorUtils.PPU_LF_ATTR, {}).get(AmountCalculatorUtils.VALUE, "0")
        qty = AmountCalculatorUtils.parse_double_or_default(qty_str, 0.0)
        price = AmountCalculatorUtils.parse_double_or_default(price_str, 0.0)
        return AmountCalculatorUtils.round_double(qty * price, 2)

    @staticmethod
    def calc_measurement_form(m: Dict, identifier: str, rate_identifier: str) -> float:
        rate_obj = m.get(rate_identifier, {})
        rate_val_str = rate_obj.get(AmountCalculatorUtils.VALUE, "0")
        rate_val = AmountCalculatorUtils.parse_double_or_default(rate_val_str, 0.0)

        measurement = m.get(identifier, {})
        length = AmountCalculatorUtils.parse_double_or_default(measurement.get(AmountCalculatorUtils.LENGTH, "0"), 0.0)
        width = AmountCalculatorUtils.parse_double_or_default(measurement.get(AmountCalculatorUtils.WIDTH, "0"), 0.0)

        rate_selected_option = rate_obj.get(AmountCalculatorUtils.SELECTED_OPTION, "")
        if rate_selected_option == AmountCalculatorUtils.PER_SQ_FT:
            selected_unit = measurement.get(AmountCalculatorUtils.SELECTED_OPTION, "")
            unit_converted = AmountCalculatorUtils.calc_measurement_unit(length * width, selected_unit)
            return AmountCalculatorUtils.round_double(unit_converted * rate_val, 2)
        else:
            return AmountCalculatorUtils.round_double(rate_val, 2)

    @staticmethod
    def calc_measurement_unit(value: float, unit: str) -> float:
        u = unit.lower()
        if u == AmountCalculatorUtils.INCHES:
            return value * AmountCalculatorUtils.INCH_TO_FEET
        elif u == AmountCalculatorUtils.MM:
            return value * AmountCalculatorUtils.MM_TO_FEET
        else:
            return value

    @staticmethod
    def round_double(value: float, places: int) -> float:
        if places < 0:
            raise ValueError("Invalid rounding scale")
        bd = Decimal(value).quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
        return float(bd)

    @staticmethod
    def calc_room_amount(items: List) -> float:
        total = sum(i.amount for i in items)
        return AmountCalculatorUtils.round_double(total, 2)

    @staticmethod
    def get_total_amount(estimator) -> float:
        amount = estimator.amount or 0.0
        discount = estimator.discount
        gst = estimator.gst
        if discount:
            amount -= (amount * discount) / 100
        if gst:
            amount += (amount * gst) / 100
        return AmountCalculatorUtils.round_double(amount, 2)

    @staticmethod
    def get_amount(rooms: List) -> float:
        total = sum(r.amount for r in rooms)
        return AmountCalculatorUtils.round_double(total, 2)

    @staticmethod
    def convert_room_attribute_json(json_str: str) -> Dict:
        try:
            parsed = json.loads(json_str or "{}")
            result = {}
            for k, v in parsed.items():
                result[k] = v if isinstance(v, dict) else {}
            return result
        except Exception as e:
            logger.error("Error parsing json: %s", json_str)
            return {}

    @staticmethod
    def parse_double_or_default(value: str, default: float) -> float:
        try:
            return float(value)
        except Exception:
            logger.error("Error parsing double: %s, defaulting to %s", value, default)
            return default
