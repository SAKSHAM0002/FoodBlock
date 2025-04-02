# Decentralised simulation
import pandas as pd
from pprint import pprint
from matplotlib import pyplot as plt
# import pickle
from collections import Counter
import geopy.distance
import simpy
import random
from math import ceil
import numpy as np
import pickle as pkl

NUM_BOYS_PER_COMPANY = 5 # 4 maybe
NUM_OF_COMPANIES = 20
NUM_BOYS = NUM_BOYS_PER_COMPANY * NUM_OF_COMPANIES # 20 maybe
BIKE_SPEED = 40
AVERAGE_RIDER_RATING = 2500
AVERAGE_CUSTOMER_RATING = 2.5
NUM_ORDERS_DONE = 0
NUM_BOYS_IN_AUCTION = ceil(NUM_BOYS/20)

K_MEANS_DIST = 4
AUCTION_K = 5
MAX_NO_OF_ORDER_RIDER_CAN_TAKE = 3    # 3,6

BOYS = []
LOG_DATA = []
ORDER_DATA = [] # distance, time
NUM_CUSTOMERS = 0
K_MEANS_DATA = []
DELIVERY_CHARGES_RATING = [] # delivery charges, customer rating for this order

MAX_LAT, MIN_LAT, MAX_LONG, MIN_LONG = 0,0,0,0

data = None

def dist(lat1, long1, lat2, long2):
    # returns the distance between two points by 
    # shortest distance on surface of earth
    return geopy.distance.geodesic((lat1, long1), (lat2, long2)).km

def man_dist(lat1, long1, lat2, long2):
    return dist(lat1, long1, lat2, long1) + dist(lat2, long1, lat2, long2)

def min_bid (order):
    return ceil(order * 0.1)

def max_bid (order):
    return ceil(order * 0.2)

def dist_travelled(rider_ind, lat, long, client_lat, client_long):
    global BOYS

    total_dist = 0
    total_dist += ceil(man_dist(BOYS[i]['lat'], BOYS[i]['long'], lat, long))
    total_dist += ceil(man_dist(lat, long, client_lat, client_long))

    return total_dist

def machine_predicted_bid(total_dist, order_cost):
    # rider_ind is the index of the delivery boy in the BOYS array
    
    min_bid_res = min_bid(order_cost)
    max_bid_res = max_bid(order_cost)

    return ceil((max_bid_res - min_bid_res)*0.5*total_dist/40) + min_bid_res
    
def first_bid(total_dist, order_cost):
    # rider_ind is the index of the delivery boy in the BOYS array
    
    min_bid_res = min_bid(order_cost)
    max_bid_res = max_bid(order_cost)

    return ceil((max_bid_res - min_bid_res)*1.0*total_dist/40) + min_bid_res
    
def second_bid(total_dist, order_cost):
    # rider_ind is the index of the delivery boy in the BOYS array
    
    min_bid_res = min_bid(order_cost)
    max_bid_res = max_bid(order_cost)

    return ceil((max_bid_res - min_bid_res)*0.75*total_dist/40) + min_bid_res

def third_bid(total_dist, order_cost):
    # rider_ind is the index of the delivery boy in the BOYS array
    
    min_bid_res = min_bid(order_cost)
    max_bid_res = max_bid(order_cost)

    return ceil((max_bid_res - min_bid_res)*0.5*total_dist/40) + min_bid_res

def customer_rating(actual_wait_time, min_wait_time):
    '''
    actual_wait_time is the time that the customer had to wait to receive the order after placing the order
    min_wait_time is the time required for the delivery boy to travel from restaurant to client location
    '''

    # here 5-1 in the formula specifies that when wait_time is 5 times the minimum possible, the rating is 2.5
    return 5*(np.exp(-((actual_wait_time-min_wait_time)/min_wait_time/(5-1)*np.log(2))))

def update_rider_rating(customer_rating, rider_ind):
    global AVERAGE_RIDER_RATING, AVERAGE_CUSTOMER_RATING, NUM_ORDERS_DONE, BOYS

    older_rating = BOYS[rider_ind]['rating']

    e = 1/(1+10**((AVERAGE_RIDER_RATING-older_rating)/800))
    change = 32*((1 if customer_rating > AVERAGE_CUSTOMER_RATING*0.8 else 0)-e)

    BOYS[rider_ind]['rating'] = older_rating + change

    AVERAGE_CUSTOMER_RATING = (AVERAGE_CUSTOMER_RATING*NUM_ORDERS_DONE + customer_rating)/(NUM_ORDERS_DONE + 1)
    NUM_ORDERS_DONE += 1

    AVERAGE_RIDER_RATING = ((NUM_BOYS*AVERAGE_RIDER_RATING) + change)/NUM_BOYS

def simulate_restaurant_owner(bids_df, order_cost):
    
    k = (order_cost-500)*(5000-0)/(3000-500) + 0

    bids_df['selection_prob'] = np.exp(-(((bids_df['rider_rating']-k)/2000)**2))
 
    return bids_df

def generate_bikers_ind_list(lat,long,num,time_now):
    reach_by = []
    # array that will store index of all riders, and the time in simulation by which they can reach the restaurant given

    for i in range(NUM_BOYS):
        time_to_reach = BOYS[i]['free_at']
        if(time_to_reach <=time_now):
            time_to_reach = time_now

        time_to_reach += ceil(man_dist(BOYS[i]['lat'], BOYS[i]['long'], lat, long)/BIKE_SPEED*60)
        reach_by.append((i, time_to_reach))

    reach_by.sort(key=lambda x:x[1])

    result = []
    for i in range(num):
        result.append(reach_by[i][0])

    return result

def perform_batch_auction(riders_ind_list, orders):     # orders is a list where each element is a tuple that stores - (order_num, res_lat, res_long,client_lat,client_long, order_cost)
    
    bids_df_list = []                                   # bids_df_list is a list that contains pandas_data_frames(each data_frame stores information regarding each order in the batch_size) as its elements
    # delivery_charges = []                                # array to store the delivery charge of all the orders
    
    for order in orders:
        order_num = order['order_num']
        lat = order['res_lat']
        long = order['res_long']
        client_lat = order['client_lat']
        client_long = order['client_long']
        order_cost = order['order_cost']

        # print(order_cost)

        min_bid_res = min_bid(order_cost)
        max_bid_res = max_bid(order_cost)
        machine_predicted_bid_res = []
        first_bid_res = []
        second_bid_res = [] 
        third_bid_res = []
        ratings = []
        rider_counts = Counter()

        for i in riders_ind_list:
            curr_dist = dist_travelled(i, lat, long, client_lat, client_long)
            machine_predicted_bid_res.append(int(machine_predicted_bid(curr_dist, order_cost)))
            first_bid_res.append(int(first_bid(curr_dist, order_cost)))
            second_bid_res.append(int(second_bid(curr_dist, order_cost)))
            third_bid_res.append(int(third_bid(curr_dist, order_cost)))
            ratings.append(BOYS[i]['rating'])
        

    

        bids = {
            'rider_ind' : riders_ind_list, 
            'machine_predicted_bid': machine_predicted_bid_res,
            'first_bid': first_bid_res, 
            'second_bid': second_bid_res, 
            'third_bid': third_bid_res,
            'rider_rating': ratings
        }
    
        bids_df = pd.DataFrame(bids)
        bids_df.set_index('rider_ind', drop=False, inplace=True)
        bids_df = simulate_restaurant_owner(bids_df, order_cost)
        bids_df_list.append(bids_df)

    for bids_df in bids_df_list:
        chosen_rider = random.choices(list(bids_df['rider_ind']), weights=list(bids_df['selection_prob']))
        rider_counts[chosen_rider[0]] += 1

    max_count = max(rider_counts.values())
    candidate_riders = [rider for rider, count in rider_counts.items() if count == max_count]

    final_rider = candidate_riders[0]

    for order, bids_df in zip(orders, bids_df_list):
        if final_rider in bids_df['rider_ind'].values:
            final_rider_bids = bids_df[bids_df['rider_ind'] == final_rider]
            final_bid = final_rider_bids['third_bid'].tolist()
            order['delivery_charge'] = final_bid
            # delivery_charges.append(third_bid)

    return final_rider, orders

def get_index_of_nearest_boy(lat,long,time_now,order_num):
    min_time = 1000000000
    min_ind = -1
    dist_corresponding_to_min = 1000000000
    
    global K_MEANS_DATA
    nearby_bikers = 0

    for i in range(NUM_BOYS):
        new_time = BOYS[i]['free_at']
        if(BOYS[i]['free_at'] <= time_now):
            new_time = time_now
        dist = man_dist(lat, long, BOYS[i]['lat'], BOYS[i]['long'])

        new_time += ceil((dist/BIKE_SPEED)*60)

        if(new_time-time_now<=ceil((K_MEANS_DIST/BIKE_SPEED)*60)):
           nearby_bikers += 1
           
        if(min_time > new_time):
            min_ind = i
            min_time = new_time
            dist_corresponding_to_min = dist

    K_MEANS_DATA.append((order_num, nearby_bikers))
    return min_ind, min_time, dist_corresponding_to_min

def save_data(curr_time):
    if(len(LOG_DATA)==0):
        LOG_DATA.append((curr_time, NUM_CUSTOMERS))
    elif (LOG_DATA[-1][0] == curr_time):
        LOG_DATA[-1] = (LOG_DATA[-1][0], NUM_CUSTOMERS)
    else:
        time = LOG_DATA[-1][0]
        val = LOG_DATA[-1][1]
        while(LOG_DATA[-1][0] < curr_time-1):
            time += 1
            LOG_DATA.append((time, val))
            # print('.', end='')
        LOG_DATA.append((curr_time, NUM_CUSTOMERS))
        # print()

def nearest_neighbour_distance(orders):
    current_lat = orders[0]['res_lat']
    current_long = orders[0]['res_long']

    distances = []
    sorted_orders = []
    unvisited_orders = orders.copy()
    while unvisited_orders:
        nearest_order = None
        nearest_distance = float('inf')

        for order in unvisited_orders:
            distance = man_dist(current_lat, current_long, order['client_lat'], order['client_long'])
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_order = order

        distances.append(nearest_distance)
        sorted_orders.append(nearest_order)

        current_lat = nearest_order['client_lat']
        current_long = nearest_order['client_long']

        unvisited_orders.remove(nearest_order)

    return distances, sorted_orders





data, dataset_acronym = pd.read_pickle(r'data\mumbai_all_in_one_day_multiple.pkl'), 'ONEDAY'
pd.options.display.max_rows = None
pd.options.display.max_columns = 4

a = 'Restaurant_latitude'
b = 'Delivery_location_latitude'
c = 'Restaurant_longitude'
d = 'Delivery_location_longitude'

MAX_LAT = max(data[a].max(), data[b].max())
MIN_LAT = min(data[a].min(), data[b].min())
MAX_LONG = max(data[c].max(), data[d].max())
MIN_LONG = min(data[c].min(), data[d].min())
N = len(data)




for i in range(NUM_BOYS):
    BOYS.append({'lat':random.randrange(10001), 'long':random.randrange(10001)})

for i in range(NUM_BOYS):
    BOYS[i]['lat'] = MIN_LAT + (MAX_LAT - MIN_LAT)*BOYS[i]['lat']/10000
    BOYS[i]['long'] = MIN_LONG + (MAX_LONG - MIN_LONG)*BOYS[i]['long']/10000
    BOYS[i]['free_at'] = 0
    BOYS[i]['num_rides'] = 0
    BOYS[i]['rating'] = 2500

''' free_in explanation
if BOYS['free_in'] == 0:
    then it means that the location of the boy is at it lat, long. else if the free_in has some value, it means that at free_at time in the simulation environment in minutes, the boy will become free, and then it's location will become equal to lat,long which is currently stored in it's entry

to check if boy is free or not, you can simply check whether his free_at is less than or equal to current time of simulation or not. if yes, he is free, else he is busy right now and will become free at his free_at time.

free_at means boy is free at the starting edge of that minute
'''

''' data.columns print, min_dist, max_dist, N, freq per day
0 ID
1 Restaurant_latitude
2 Restaurant_longitude
3 Delivery_location_latitude
4 Delivery_location_longitude
5 Order_Date
6 Time_Orderd
7 Time_Order_picked
8 Time_taken(min)
9 order_pick
10 order_place
11 order_delivered

N = 2872
min_dist = 0.05763515594883869
max_dist = 66.00012665542367

{datetime.date(2022, 4, 1): 375,
 datetime.date(2022, 4, 2): 429,
 datetime.date(2022, 4, 3): 408,
 datetime.date(2022, 4, 4): 531,
 datetime.date(2022, 4, 5): 333,
 datetime.date(2022, 4, 6): 467,
 datetime.date(2022, 4, 7): 329}

'''

class Restaurant:
    def __init__(self, env, boys, name, restaurant_orders):
        self.env = env
        self.boys = boys
        self.name = name
        self.restaurant_orders = restaurant_orders
        self.res_lat = restaurant_orders[0]['res_lat']
        self.res_long = restaurant_orders[0]['res_long']
        
        self.route = None
        self.bike_ind = None
        self.bike_reach_restaurant_at = None
        self.start_time = 0
        
        self.delivery_charges = None

    def action(self):
        global NUM_CUSTOMERS, ORDER_DATA, NUM_BOYS_IN_AUCTION, DELIVERY_CHARGES_RATING

        NUM_CUSTOMERS += len(self.restaurant_orders)
        # NUM_CUSTOMERS += 1

        save_data(self.env.now)

        self.start_time = self.env.now

        # self.bike_ind, self.bike_reach_restaurant_at, dist1 = get_index_of_nearest_boy(self.res_lat, self.res_long, self.env.now, self.order_num)

        self.bike_ind, self.restaurant_orders = perform_batch_auction(generate_bikers_ind_list(self.res_lat, self.res_long, NUM_BOYS_IN_AUCTION, self.env.now), self.restaurant_orders)

        dist1 = man_dist(self.res_lat, self.res_long, BOYS[self.bike_ind]['lat'], BOYS[self.bike_ind]['long'])
        self.bike_reach_restaurant_at = BOYS[self.bike_ind]['free_at']
        
        if(self.bike_reach_restaurant_at < self.env.now):
            self.bike_reach_restaurant_at = self.env.now
        
        self.bike_reach_restaurant_at += ceil((dist1/BIKE_SPEED)*60)

        yield env.timeout(self.bike_reach_restaurant_at - self.env.now)

        self.route, self.restaurant_orders = nearest_neighbour_distance(self.restaurant_orders)

        BOYS[self.bike_ind]['free_at'] += ((sum(self.route) / BIKE_SPEED) * 60)

        # order_dist = 0

        for i,distance in enumerate(self.route):
            # order_dist += distance
            order_details = self.restaurant_orders[i]
            delivery_time = ceil((distance / BIKE_SPEED)*60)

            BOYS[self.bike_ind]['lat'] = order_details['client_lat']
            BOYS[self.bike_ind]['long'] = order_details['client_long']

            yield self.env.timeout(delivery_time)

            NUM_CUSTOMERS -= 1
            save_data(self.env.now)
            BOYS[self.bike_ind]['num_rides'] += 1
            actual_wait_time = self.env.now - self.start_time
            ORDER_DATA.append((dist1 + distance, actual_wait_time))
            rating_by_customer = customer_rating(actual_wait_time, ((dist1/BIKE_SPEED)*60) + delivery_time)
            DELIVERY_CHARGES_RATING.append((order_details['delivery_charge'], rating_by_customer))
            update_rider_rating(rating_by_customer, self.bike_ind)

            print(f"{order_details['order_num']} ", end='',flush=True)



def restaurant_generator(env, boys):
    global data, N, MAX_NO_OF_ORDER_RIDER_CAN_TAKE

    encountered_orders = set()  # a set to keep track of encountered order ID's
    unique_group_keys = data['order_place'].unique()

    for i in range(len(unique_group_keys)):
        current_group_key = unique_group_keys[i]
        group_data = data[data['order_place'] == current_group_key]

        if i == 0:
            t = 0
        else:
            previous_group_key = unique_group_keys[i - 1]
            t = (pd.to_datetime(current_group_key) - pd.to_datetime(previous_group_key)).total_seconds() / 60

        yield env.timeout(ceil(t))

        # Iterate over the group data to process orders
        for index, row in group_data.iterrows():
            order_id = row['ID']

            if order_id in encountered_orders:
                continue

            restaurant_orders = []  # List to store orders from the same restaurant in a batch
            for _, row_in_group in group_data.iterrows():
                if (
                    row_in_group['Restaurant_location'] == row['Restaurant_location']
                    and len(restaurant_orders) <= MAX_NO_OF_ORDER_RIDER_CAN_TAKE
                    and row_in_group['ID'] not in encountered_orders
                ):
                    restaurant_orders.append({
                        'order_num': row_in_group.name,
                        'res_lat': row_in_group['Restaurant_latitude'],
                        'res_long': row_in_group['Restaurant_longitude'],
                        'client_lat': row_in_group['Delivery_location_latitude'],
                        'client_long': row_in_group['Delivery_location_longitude'],
                        'order_cost': row_in_group['order_cost']
                    })

                    encountered_orders.add(row_in_group['ID'])

            if restaurant_orders:  # Ensure there's at least one order to process
                r = Restaurant(
                    env=env,
                    boys=boys,
                    name=f"Restaurant {row['Restaurant_name']}",
                    restaurant_orders=restaurant_orders
                )
                env.process(r.action())


        


env = simpy.Environment()
boys = simpy.Resource(env, NUM_BOYS)
env.process(restaurant_generator(env, boys))

env.run()



x = []
y = []

for i in LOG_DATA:
    # print(i)
    x.append(i[0])
    y.append(i[1])

x = np.array(x)
y = np.array(y)

with open(f"data/{dataset_acronym}_NUM_BOYS_{NUM_BOYS}_BIKE_SPEED_{BIKE_SPEED}_NUM_BOYS_PER_COMPANY_{NUM_BOYS_PER_COMPANY}_NUM_OF_COMPANIES_{NUM_OF_COMPANIES}_decent_multiple.pkl", 'wb') as file:
    pkl.dump((x,y), file)

with open(f"data/{dataset_acronym}_NUM_BOYS_{NUM_BOYS}_BIKE_SPEED_{BIKE_SPEED}__NUM_BOYS_PER_COMPANY_{NUM_BOYS_PER_COMPANY}_NUM_OF_COMPANIES_{NUM_OF_COMPANIES}_ORDER_DATA.pkl_multiple", 'wb') as file:
    pkl.dump(ORDER_DATA, file)

with open(f"data/{dataset_acronym}_NUM_BOYS_{NUM_BOYS}_BIKE_SPEED_{BIKE_SPEED}__NUM_BOYS_PER_COMPANY_{NUM_BOYS_PER_COMPANY}_NUM_OF_COMPANIES_{NUM_OF_COMPANIES}_DELIVERY_CHARGES_RATING_multiple.pkl", 'wb') as file:
    pkl.dump(DELIVERY_CHARGES_RATING, file)

with open(f"data/{dataset_acronym}_NUM_BOYS_{NUM_BOYS}_BIKE_SPEED_{BIKE_SPEED}__NUM_BOYS_PER_COMPANY_{NUM_BOYS_PER_COMPANY}_NUM_OF_COMPANIES_{NUM_OF_COMPANIES}_K_DIST_{K_MEANS_DIST}_K_MEANS_DATA_multiple.pkl", 'wb') as file:
    pkl.dump(K_MEANS_DATA, file)




plt.figure('Queue Length vs Time Decentralised One company')
plt.title(f"NUM_BOYS = {NUM_BOYS}, NUM_BOYS_PER_COMPANY = {NUM_BOYS_PER_COMPANY}, BIKE_SPEED = {BIKE_SPEED}, NUM_OF_COMPANIES = {NUM_OF_COMPANIES}")
plt.plot(x,y, color='#fc5c65')

plt.axhline(np.average(y),0, np.amax(x), color='#576574')
print(f'\nDecentralised average queue length is {np.average(y)}\nNUM_BOYS_PER_COMPANY = {NUM_BOYS_PER_COMPANY}\n')

plt.show()

# print(f'BOYS : {BOYS}')
# print(f'LOG DATA : {LOG_DATA}')
print(f'ORDER DATA : {ORDER_DATA}') 
order_data_length = len(ORDER_DATA)
print(f'Length of order data {order_data_length}')
# print(f'NUM CUSTOMERS : {NUM_CUSTOMERS}')
# print(f'K MEANS DATA : {K_MEANS_DATA}')
# print(f'DELIVERY CHARGES RATING : {DELIVERY_CHARGES_RATING}') 



average_rating_boys = sum(boy['rating'] for boy in BOYS) / len(BOYS)
average_delivery_rating = sum(rating_tuple[1] for rating_tuple in DELIVERY_CHARGES_RATING) / len(DELIVERY_CHARGES_RATING)

print(f'Average rating of rider: {average_rating_boys}')
print(f'Average rating given by customer: {average_delivery_rating}')
print(AVERAGE_RIDER_RATING)
print(AVERAGE_CUSTOMER_RATING)


distances = [order[0] for order in ORDER_DATA]
average_distance = sum(distances) / len(distances)

wait_times = [order[1] for order in ORDER_DATA]
average_wait_time = sum(wait_times) / len(wait_times)

print(f'Average distance travelled: {average_distance} km')
print(f'Average wait time: {average_wait_time} minutes')
print('Number of orders done : {NUM_ORDERS_DONE}')
print(NUM_ORDERS_DONE)




