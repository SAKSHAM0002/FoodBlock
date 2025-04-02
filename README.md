# FoodBlock
A Secure and Cost-Optimal Framework for Online Food Ordering

NOTE-
The following explanations illustrate how the functions work together.

1)def restaurant_generator - 

Takes two arguments - simulation environment and boys(list of delivery boys).
encountered_orders is a set that contains a track of encountered order_ID's.
unique_group_keys is an array that contains different time slots when orders are placed.
Looping over all the time slots:
	current_group_key has the current time slot
	group_data has all the rows from data where order_place is the current_group_key.
	
	for the first iteration do not pause the simulation
	second iteration onwards pause the simulation for the duration that is equal to the difference between current time slot and the previous time slot.

	Looping over the rows in the current time slot:
		index is the index of the current row
		row is the series object containing the row corresponding to that index
		
		if the current order has already been processed,continue(go to the next row)

		Store all the orders that are from the same restaurant location as that of the current row restaurant location
		
		All the orders that have been stored in the restaurant_orders array
		Mark all those orders as encountered

Process all the restaurant_orders together using the Restaurant class method.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------


2)def perform_batch_auction - 

takes two arguments rider_index_list(list of riders) and orders(orders is a list where each element is a tuple that stores - (order_num, res_lat, res_long,client_lat,client_long, order_cost))

bids_df_list is a list that contains pandas_data_frames(each data_frame stores information regarding each order) as its elements

Looping through orders:
	For 1st order find the machine_prdected of all the riders
	For 1st order find the first_bid of all the riders
	For 1st order find the second_bid of all the riders
	For 1st order find the third_bid of all the riders
	Also initialise an array to store the rider_rating

	Make a dataframe for the first order having the following coloumns:rider_ind,machine_predicted_bid,first_bid,second_bid,third_bid,rider_rating

	For the first order dataframe add additional column 'selection_probability' using simulate_restaurant_owner funtion	

	Append this dataframe into bits_df_list

For every order(dataframe) chose a rider using random.choices and selection probability as weights
Chose a rider for every order
Chose the rider for the delivery of whole batch which is chosen for the most number of times
If there are riders who are chosen equal number of times, select the very first rider_index since he will reach the restaurant the earliest(since the list provided is already arranged in ascending order of reaching_time to restaurant).


For every order add another key_value pair in the orders list - delivery_charge corresponding to the final rider

-------------------------------------------------------------------------------------------------------------------------------------------------------------------


3)def action

Increment the number of customer to the number of orders in the restaurant_orders
save the data

start the clock

Find the rider and all the delivery charge for the orders in restaurant_orders using perform_batch auction

find the time at which that rider will reach the restaurant
pause the simulation until that time

Find the sequence in which the rider will deliver the orders in restaurant_order using nearest neighbour algorithm

route is an array contains - [distance between restaurant and first order location, distance between first order location and second order location,distance between second order location and third order location,......]
restaurant_orders after calling nearest neighbour distance function will now contain all the restaurant orders in the order in which they have to be delivered 

Now the rider will be busy until he delivers the last order(Update 'free_at' to the time at which he will deliver the last order)

Now the rider will deliver each order one by one based on the sorted restaurant orders
the route array that contains distances between two delivery locations is used to calculate the wait times.

As soon as the rider is going to deliver the first order, update the rider's coordinate to the first client's location

pause the simulation until he delivers the order

After delivery
Decrement the number of customers by 1
save data
Increment the number of rides done be riders by 1
Update ORDER_DATA
UPDATE DELIVERY_CHARGES_RATING

Go to the next iteration

-------------------------------------------------------------------------------------------------------------------------------------------------------------------


NOTE:

A)Nearest neighbour algorithm used - 
Find the next nearest location and travel to that location
Then from that location find the next nearest location and travel to it


B)If the rider has to deliver 3 orders - order 1, order 2, order 3
  And the rider will deliver the order in the sequence - order 2, order 1 , order 3

  1st iteration - 
	Rider will go from restaurant to the 2nd client's location
	For order 2 the total distance travelled is the sum of (distance between the restaurant and the 2nd client's location and the distance travelled by the rider to reach restaurant)

  2nd iteration -
	Rider will go from 2nd client's location to the 1st client's location
	For order 1 the total distance travelled is the sum of (distance between the 2nd client's location and the 1st client's location and distance travelled by the rider to reach the 	restaurant only ( For calculating the total distance travelled by the rider for order 1 , we do not take into account the distance he has travelled previously, i.e. the distance 	rider has travelled to deliver the previous order)

  3rd iteration - 
	Rider will go from 1st client's location to the 3rd client's location
	For order 3 the total distance travelled is the sum of (distance between the 1st client's location and the 3rd client's location and distance travelled by the rider to reach the 	restaurant only( For calculating the total distance travelled by the rider for order 3 , we do not take into account the distance he has travelled previously, i.e. the distance 	rider has travelled to deliver order 1


	


	
