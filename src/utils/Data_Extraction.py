# python
##Use the contents below in the django docker to grab the Data from the postgres qsl
from datetime import date
from django.db.models import Sum
from django.db.models.functions import TruncDate
from apps.direct_tips.models import Location, LocationDailySales, ShiftTotals
import os
import csv
from django.utils import timezone


## TODO - Accept date range as input paramenters & file name
#uncomment it for Test data
today = date(2025, 9, 1)
start_of_year = date(2024,5,1)

#uncomment it for Train data
#today = date.today()
#start_of_year = date(2025,9,1)

all_locations = Location.objects.filter(is_simple_onboarding=False, is_active=True).exclude(name__iexact='Updated design Test location')
locations = []

for l in all_locations:
    if not l.metadata.get("allow_event_shifts") and not l.metadata.get("is_comparison_location"):
        locations.append({
            'id': l.id,
            'name': l.name,
            'nickname': l.nickname,
            'city': l.city.lower(),
            'province': l.province,
            'event_location_id': l.event_location.id if l.event_location else None
        })

# get sorted list of days present in the data
days = (
    LocationDailySales.objects
    .filter(
        shift__business_date__gte=start_of_year,
        shift__business_date__lte=today,
        shift__payments_paused=False,
        shift__shift_period='',
        shift__is_simple_onboarding=False
    )
    .annotate(day=TruncDate('shift__business_date', tzinfo=timezone.utc))
    .values_list('day', flat=True)
    .distinct()
)
days = sorted(set(days))

# build sales data per location per day using location ids
sales_data = {}
for loc in locations:
    loc_id = loc['id']
    loc_sales_qs = (
        LocationDailySales.objects
        .filter(
            shift__location_id=loc_id,
            shift__business_date__gte=start_of_year,
            shift__business_date__lte=today,
            shift__payments_paused=False,
            shift__shift_period='',
            shift__is_simple_onboarding=False
        )
        .annotate(day=TruncDate('shift__business_date', tzinfo=timezone.utc))
        .values('day')
        .annotate(total_sales=Sum('net_sales'))
    )
    day_sales = {entry['day'].strftime('%Y-%m-%d'): entry['total_sales'] for entry in loc_sales_qs}
    sales_data[loc_id] = day_sales

adjusted_sales = {}
for loc in locations:
    loc_id = loc['id']
    base_day_sales = sales_data.get(loc_id, {})
    event_loc_id = loc.get('event_location_id')
    if event_loc_id:
        event_loc_sales_qs = (
            LocationDailySales.objects
            .filter(
                shift__location_id=event_loc_id,
                shift__business_date__gte=start_of_year,
                shift__business_date__lte=today,
                shift__payments_paused=False,
                shift__shift_period='',
                shift__is_simple_onboarding=False
            )
            .annotate(day=TruncDate('shift__business_date', tzinfo=timezone.utc))
            .values('day')
            .annotate(total_sales=Sum('net_sales'))
        )
        event_day_sales = {entry['day'].strftime('%Y-%m-%d'): entry['total_sales'] for entry in event_loc_sales_qs}
    else:
        event_day_sales = {}

    all_days = set(base_day_sales.keys()) | set(event_day_sales.keys())
    loc_adjusted = {}
    for d in sorted(all_days):
        base = base_day_sales.get(d)
        ev = event_day_sales.get(d)
        if base is None:
            loc_adjusted[d] = None
        elif ev is None:
            loc_adjusted[d] = base
        else:
            loc_adjusted[d] = base - ev

    adjusted_sales[loc_id] = loc_adjusted

tips_data = {}
for loc in locations:
    loc_id = loc['id']
    loc_tips_qs = (
        ShiftTotals.objects.filter(
            shift__location_id=loc_id,
            shift__business_date__gte=start_of_year,
            shift__business_date__lte=today,
            shift__payments_paused=False,
            shift__shift_period='',
            shift__is_simple_onboarding=False
        )
        .annotate(day=TruncDate('shift__business_date', tzinfo=timezone.utc))
        .values('day')
        .annotate(total_tips=Sum('total_tips'))
    )
    day_tips = {entry['day'].strftime('%Y-%m-%d'): entry['total_tips'] for entry in loc_tips_qs}
    tips_data[loc_id] = day_tips

adjusted_tips = {}
for loc in locations:
    loc_id = loc['id']
    base_day_tips = tips_data.get(loc_id, {})
    event_loc_id = loc.get('event_location_id')
    if event_loc_id:
        event_loc_tips_qs = (
            ShiftTotals.objects.filter(
                shift__location_id=event_loc_id,
                shift__business_date__gte=start_of_year,
                shift__business_date__lte=today,
                shift__payments_paused=False,
                shift__shift_period='',
                shift__is_simple_onboarding=False
            )
            .annotate(day=TruncDate('shift__business_date', tzinfo=timezone.utc))
            .values('day')
            .annotate(total_tips=Sum('total_tips'))
        )
        event_day_tips = {entry['day'].strftime('%Y-%m-%d'): entry['total_tips'] for entry in event_loc_tips_qs}
    else:
        event_day_tips = {}

    all_days = set(base_day_tips.keys()) | set(event_day_tips.keys())
    loc_adjusted = {}
    for d in sorted(all_days):
        base = base_day_tips.get(d)
        ev = event_day_tips.get(d)
        if base is None:
            loc_adjusted[d] = None
        elif ev is None:
            loc_adjusted[d] = base
        else:
            loc_adjusted[d] = base - ev

    adjusted_tips[loc_id] = loc_adjusted

# Create city_id mapping once
city_names = sorted({(loc.get('city') or '').strip().lower() for loc in locations if loc.get('city')})
city_map = {city: idx for idx, city in enumerate(city_names, start=0)}


folder_name = 'location_data'
os.makedirs(folder_name, exist_ok=True)
filename = os.path.join(folder_name, 'daily_tips_sales_train.csv')

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['date', 'location_id', 'location_name', 'location_nickname', 'location_city', 'city_id', 'province/state', 'daily_sales', 'daily_tips'])
    for loc in locations:
        loc_id = loc['id']
        city_id = city_map.get(loc.get('city'), "")
        for day in days:
            day_str = day.strftime('%Y-%m-%d')
            sales = adjusted_sales.get(loc_id, {}).get(day_str, "Missing Data")
            tips = adjusted_tips.get(loc_id, {}).get(day_str, "Missing Data")
            writer.writerow([day_str, loc_id, loc['name'], loc['nickname'], loc['city'], city_id, loc['province'], sales, tips])

with open('locations.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['location_id', 'location_name', 'location_nickname', 'location_city', 'city_id', 'province/state'])
    for loc in locations:
        loc_id = loc['id']
        city_id = city_map.get(loc.get('city'), "")
        writer.writerow([loc_id, loc['name'], loc['nickname'], loc['city'], city_id, loc['province']])

with open('city.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['city_name', 'city_id'])
    for city, cid in sorted(city_map.items(), key=lambda kv: kv[1]):
        writer.writerow([city, cid])