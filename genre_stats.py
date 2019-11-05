import numpy as np
import pandas as pd
import sys
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Need to decode this at utf-8 or else python complains about encoding mappings in the underlying pandas functions
genres = pd.read_json('genres.json.gz', orient= 'record', lines = True, encoding='utf-8')
wikidata = pd.read_json('wikidata-movies.json.gz', orient = 'record', lines = True, encoding='utf-8')
omdb =  pd.read_json('omdb-data.json.gz', orient = 'record', lines = True, encoding='utf-8')
rotten_t = pd.read_json('rotten-tomatoes.json.gz', orient = 'record', lines = True, encoding='utf-8')

wikidata = wikidata[~(wikidata['enwiki_title'].isnull())]


rotten_t = rotten_t.merge(omdb, on = 'imdb_id')
rotten_t = rotten_t.merge(wikidata, on = 'rotten_tomatoes_id')

#Clean the data even more to remove null values that may be in the audience average or percentage
rotten_t = rotten_t[ ~(rotten_t['critic_average'].isnull()) ]
rotten_t = rotten_t[ ~(rotten_t['audience_average'].isnull()) ]
rotten_t = rotten_t[ ~(rotten_t['critic_percent'].isnull()) ]
rotten_t = rotten_t[ ~(rotten_t['audience_percent'].isnull()) ]

#Making the audience average out of 10 just like the critic average (normalization step)
rotten_t['audience_average'] = rotten_t['audience_average'] * 2

#Create all our different genre dataframes by looking for keywords
action_df = rotten_t[['Action' in x for x in rotten_t['omdb_genres']]]
action_df = action_df[['omdb_genres', 'audience_average', 'audience_ratings', 'critic_average', 'critic_percent', 'audience_percent', 'label', 'omdb_awards', 'publication_date']]

comedy_df = rotten_t[['Comedy' in x for x in rotten_t['omdb_genres']]]
comedy_df = comedy_df[['omdb_genres', 'audience_average', 'audience_ratings', 'critic_average', 'critic_percent', 'audience_percent', 'label', 'omdb_awards']]

fantasy_df = rotten_t[['Fantasy' in x for x in rotten_t['omdb_genres']]]
fantasy_df = fantasy_df[['omdb_genres', 'audience_average', 'audience_ratings', 'critic_average', 'critic_percent', 'audience_percent', 'label', 'omdb_awards']]

drama_df = rotten_t[['Drama' in x for x in rotten_t['omdb_genres']]]
drama_df = drama_df[['omdb_genres', 'audience_average', 'audience_ratings', 'critic_average', 'critic_percent', 'audience_percent', 'label', 'omdb_awards']]


#Histograms for normal looking data and well over n>40 data points each.
#As we can see by the outputs audience/critic averages are a lot more normally distributed than the percentages with drama genre percentages being
#largely left skewed in both audience and critic headings
plt.subplot(2,2,1)
plt.hist(action_df['audience_average'], color='r')
plt.title('Audience Scores: Action')
plt.subplot(2,2,2)
plt.hist(comedy_df['audience_average'], color='y')
plt.title('Audience Scores: Comedy')
plt.subplot(2,2,3)
plt.hist(fantasy_df['audience_average'], color='m')
plt.title('Audience Scores: Fantasy')
plt.subplot(2,2,4)
plt.hist(drama_df['audience_average'], color='c')
plt.title('Audience Scores: Drama')
plt.savefig('Audience Scores hist plt')
plt.close()

plt.subplot(2,2,1)
plt.hist(action_df['critic_average'], color='r')
plt.title('Critic Scores: Action')
plt.subplot(2,2,2)
plt.hist(comedy_df['critic_average'], color='y')
plt.title('Critic Scores: Comedy')
plt.subplot(2,2,3)
plt.hist(fantasy_df['critic_average'], color='m')
plt.title('Critic Scores: Fantasy')
plt.subplot(2,2,4)
plt.hist(drama_df['critic_average'], color='c')
plt.title('Critic Scores: Drama')
plt.savefig('Critic Scores hist plt')
plt.close()


plt.subplot(2,2,1)
plt.hist(action_df['audience_percent'], color='r')
plt.title('Audience Percent: Action')
plt.subplot(2,2,2)
plt.hist(comedy_df['audience_percent'], color='y')
plt.title('Audience Percent: Comedy')
plt.subplot(2,2,3)
plt.hist(fantasy_df['audience_percent'], color='m')
plt.title('Audience Percent: Fantasy')
plt.subplot(2,2,4)
plt.hist(drama_df['audience_percent'], color='c')
plt.title('Audience Percent: Drama')
plt.savefig('Audience Percent hist plt')
plt.close()

plt.subplot(2,2,1)
plt.hist(action_df['critic_percent'], color='r')
plt.title('Critic Percent: Action')
plt.subplot(2,2,2)
plt.hist(comedy_df['critic_percent'], color='y')
plt.title('Critic Percent: Comedy')
plt.subplot(2,2,3)
plt.hist(fantasy_df['critic_percent'], color='m')
plt.title('Critic Percent: Fantasy')
plt.subplot(2,2,4)
plt.hist(drama_df['critic_percent'], color='c')
plt.title('Critic Percent: Drama')
plt.savefig('Critic Percent hist plt')
plt.close()

#This code just checks the histogram of our transformed drama percentages but it did not help the skewedness
#This also showed the original data average histograms before
'''
#Square rooting makes our p-values not work on drama and does not fix the left skew on our data
#logarithithm doesn't allow us to plot a histogram and makes some numbers NAN which is not what we want i.e. transforming data fails 
drama_df['critic_percent'] = np.sqrt(drama_df['critic_percent'])
drama_df['audience_percent'] = np.sqrt(drama_df['audience_percent'])
plt.subplot(1,2,1)
plt.hist(drama_df['critic_percent'], color='c')
plt.subplot(1,2,2)
plt.hist(drama_df['audience_percent'], color='c')
plt.show()

plt.subplot(1,2,1)
plt.hist(rotten_t['audience_average'])
plt.subplot(1,2,2)
plt.hist(rotten_t['critic_average'])
plt.show()
'''

#Start the data statistical analysis for critic and audience averages between genres
print('Average Critic Score for All movie genres: ', round(rotten_t['critic_average'].mean(), 2), '/10')
print('Average Critic Score for Action movies: ', round(action_df['critic_average'].mean(), 2), '/10')
print('Average Critic Score for Comedy movies: ', round(comedy_df['critic_average'].mean(), 2), '/10')
print('Average Critic Score for Drama movies: ', round(drama_df['critic_average'].mean(), 2), '/10')
print('Average Critic Score for Fantasy movies: ', round(fantasy_df['critic_average'].mean(), 2), '/10')
print('\n')

print('Critic Score P-values (less than 0.05 is significant):')
print('All genres vs Action p-value', stats.ttest_ind(rotten_t['critic_average'], action_df['critic_average']).pvalue)
print('All genres vs Comedy p-value', stats.ttest_ind(rotten_t['critic_average'], comedy_df['critic_average']).pvalue)
print('All genres vs Fantasy p-value', stats.ttest_ind(rotten_t['critic_average'], fantasy_df['critic_average']).pvalue)
print('All genres vs Drama p-value', stats.ttest_ind(rotten_t['critic_average'], drama_df['critic_average']).pvalue)
print('Action vs Fantasy p-value:', stats.ttest_ind(action_df['critic_average'], fantasy_df['critic_average']).pvalue)
print('Action vs Comedy p-value:', stats.ttest_ind(action_df['critic_average'], comedy_df['critic_average']).pvalue)
print('Action vs Drama p-value:', stats.ttest_ind(action_df['critic_average'], drama_df['critic_average']).pvalue)
print('Comedy vs Fantasy p-value:', stats.ttest_ind(comedy_df['critic_average'], fantasy_df['critic_average']).pvalue)
print('Comedy vs Drama p-value:', stats.ttest_ind(comedy_df['critic_average'], drama_df['critic_average']).pvalue)
print('Drama vs Fantasy p-value:', stats.ttest_ind(drama_df['critic_average'], fantasy_df['critic_average']).pvalue, '\n')

print('Average Audience Score for All movie genres: ', round(rotten_t['audience_average'].mean(), 2), '/10')
print('Average Audience Score for Action movies: ', round(action_df['audience_average'].mean(), 2), '/10')
print('Average Audience Score for Comedy movies: ', round(comedy_df['audience_average'].mean(), 2), '/10')
print('Average Audience Score for Drama movies: ', round(drama_df['audience_average'].mean(), 2), '/10')
print('Average Audience Score for Fantasy movies: ', round(fantasy_df['audience_average'].mean(), 2), '/10')
print('\n')

print('Audience Average P-values (less than 0.05 is significant):')
print('All genres vs Action p-value', stats.ttest_ind(rotten_t['audience_average'], action_df['audience_average']).pvalue)
print('All genres vs Comedy p-value', stats.ttest_ind(rotten_t['audience_average'], comedy_df['audience_average']).pvalue)
print('All genres vs Fantasy p-value', stats.ttest_ind(rotten_t['audience_average'], fantasy_df['audience_average']).pvalue)
print('All genres vs Drama p-value', stats.ttest_ind(rotten_t['audience_average'], drama_df['audience_average']).pvalue)
print('Action vs Fantasy p-value:', stats.ttest_ind(action_df['audience_average'], fantasy_df['audience_average']).pvalue)
print('Action vs Comedy p-value:', stats.ttest_ind(action_df['audience_average'], comedy_df['audience_average']).pvalue)
print('Action vs Drama p-value:', stats.ttest_ind(action_df['audience_average'], drama_df['audience_average']).pvalue)
print('Comedy vs Fantasy p-value:', stats.ttest_ind(comedy_df['audience_average'], fantasy_df['audience_average']).pvalue)
print('Comedy vs Drama p-value:', stats.ttest_ind(comedy_df['audience_average'], drama_df['audience_average']).pvalue)
print('Drama vs Fantasy p-value:', stats.ttest_ind(drama_df['audience_average'], fantasy_df['audience_average']).pvalue, '\n')

#Start the data statistical analysis for critic and audience percentages between genres (Remember drama dataframe is still skewed so have to take with a grain of salt)
print('Average Critic Percent for All movie genres: ', round(rotten_t['critic_percent'].mean(), 2), '/100')
print('Average Critic Percent for Action movies: ', round(action_df['critic_percent'].mean(), 2), '/100')
print('Average Critic Percent for Comedy movies: ', round(comedy_df['critic_percent'].mean(), 2), '/100')
print('Average Critic Percent for Drama movies: ', round(drama_df['critic_percent'].mean(), 2), '/100')
print('Average Critic Percent for Fantasy movies: ', round(fantasy_df['critic_percent'].mean(), 2), '/100')
print('\n')

print('Critic Percent P-values (less than 0.05 is significant):')
print('All genres vs Action p-value', stats.ttest_ind(rotten_t['critic_percent'], action_df['critic_percent']).pvalue)
print('All genres vs Comedy p-value', stats.ttest_ind(rotten_t['critic_percent'], comedy_df['critic_percent']).pvalue)
print('All genres vs Fantasy p-value', stats.ttest_ind(rotten_t['critic_percent'], fantasy_df['critic_percent']).pvalue)
print('All genres vs Drama p-value', stats.ttest_ind(rotten_t['critic_percent'], drama_df['critic_percent']).pvalue)
print('Action vs Fantasy p-value:', stats.ttest_ind(action_df['critic_percent'], fantasy_df['critic_percent']).pvalue)
print('Action vs Comedy p-value:', stats.ttest_ind(action_df['critic_percent'], comedy_df['critic_percent']).pvalue)
print('Action vs Drama p-value:', stats.ttest_ind(action_df['critic_percent'], drama_df['critic_percent']).pvalue)
print('Comedy vs Fantasy p-value:', stats.ttest_ind(comedy_df['critic_percent'], fantasy_df['critic_percent']).pvalue)
print('Comedy vs Drama p-value:', stats.ttest_ind(comedy_df['critic_percent'], drama_df['critic_percent']).pvalue)
print('Drama vs Fantasy p-value:', stats.ttest_ind(drama_df['critic_percent'], fantasy_df['critic_percent']).pvalue, '\n')

print('Average Audience Percent for All movie genres: ', round(rotten_t['audience_percent'].mean(), 2), '/100')
print('Average Audience Percent for Action movies: ', round(action_df['audience_percent'].mean(), 2), '/100')
print('Average Audience Percent for Comedy movies: ', round(comedy_df['audience_percent'].mean(), 2), '/100')
print('Average Audience Percent for Drama movies: ', round(drama_df['audience_percent'].mean(), 2), '/100')
print('Average Audience Percent for Fantasy movies: ', round(fantasy_df['audience_percent'].mean(), 2), '/100')
print('\n')

print('Audience Percent P-values (less than 0.05 is significant):')
print('All genres vs Action p-value', stats.ttest_ind(rotten_t['audience_percent'], action_df['audience_percent']).pvalue)
print('All genres vs Comedy p-value', stats.ttest_ind(rotten_t['audience_percent'], comedy_df['audience_percent']).pvalue)
print('All genres vs Fantasy p-value', stats.ttest_ind(rotten_t['audience_percent'], fantasy_df['audience_percent']).pvalue)
print('All genres vs Drama p-value', stats.ttest_ind(rotten_t['audience_percent'], drama_df['audience_percent']).pvalue)
print('Action vs Fantasy p-value:', stats.ttest_ind(action_df['audience_percent'], fantasy_df['audience_percent']).pvalue)
print('Action vs Comedy p-value:', stats.ttest_ind(action_df['audience_percent'], comedy_df['audience_percent']).pvalue)
print('Action vs Drama p-value:', stats.ttest_ind(action_df['audience_percent'], drama_df['audience_percent']).pvalue)
print('Comedy vs Fantasy p-value:', stats.ttest_ind(comedy_df['audience_percent'], fantasy_df['audience_percent']).pvalue)
print('Comedy vs Drama p-value:', stats.ttest_ind(comedy_df['audience_percent'], drama_df['audience_percent']).pvalue)
print('Drama vs Fantasy p-value:', stats.ttest_ind(drama_df['audience_percent'], fantasy_df['audience_percent']).pvalue, '\n')

#Run an anova test and against the total average dataframe and between genres.
#Alpha with 0.05 we conclude that all groups are different
nova_critic = stats.f_oneway(rotten_t['critic_average'], action_df['critic_average'], comedy_df['critic_average'], drama_df['critic_average'], fantasy_df['critic_average'] )
print("The p-value for the ANOVA between our critic averages on genres is: ")
print(nova_critic.pvalue)
nova_audience= stats.f_oneway(rotten_t['audience_average'], action_df['audience_average'], comedy_df['audience_average'], drama_df['audience_average'], fantasy_df['audience_average'])
print("The p-value for the ANOVA between our audience averages on genres is: ")
print(nova_audience.pvalue)

#Run an anova test and against the total percentage dataframe and between genres.
#Alpha with 0.05 we conclude that all groups are different
nova_critic_p = stats.f_oneway(rotten_t['critic_percent'], action_df['critic_percent'], comedy_df['critic_percent'], drama_df['critic_percent'], fantasy_df['critic_percent'] )
print("The p-value for the ANOVA between our critic percentages on genres is: ")
print(nova_critic_p.pvalue)
nova_audience_p = stats.f_oneway(rotten_t['audience_percent'], action_df['audience_percent'], comedy_df['audience_percent'], drama_df['audience_percent'], fantasy_df['audience_percent'])
print("The p-value for the ANOVA between our audience percentages on genres is: ")
print(nova_audience_p.pvalue)

'''
#Cannot run a posthoc on means for some reason.  This is being called the same way we used it before but get an error: ValueError: v must be > 1 when p >= .9
#Not sure why but google has no solutions to this error even on simple dataframes doesnt work for percentages or averages
critic_data = {'Genres': ['All', 'Action', 'Comedy', 'Fantasy', 'Drama'], 'Means': [rotten_t['critic_percent'].mean(), action_df['critic_percent'].mean(), comedy_df['critic_percent'].mean(), fantasy_df['critic_percent'].mean(), drama_df['critic_percent'].mean()]}
critic_posthoc_df = pd.DataFrame(data=critic_data)
print(critic_posthoc_df)
posthoc = pairwise_tukeyhsd(critic_posthoc_df['Means'], critic_posthoc_df['Genres'], alpha=0.05)
print(posthoc)
'''