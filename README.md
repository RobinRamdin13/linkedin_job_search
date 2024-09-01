# linkedin_job_search
This is a naive approach at identifying personalized job positing from LinkedIn using a user define set of skills for a job position. The geolocation for the job searches are based in Singapore, however it can easily be updated to the desired geolocation. 

### Overview
The code consists of two parts: 
- **LinkedIn Job Extraction:** &#8594; This part will use an API call to LinkedIn using the parameters defined in `config.yaml` to extract a list of jobs
- **Finding Best Suited Job:** &#8594; Using FAISS to index the extracted information from LinkedIn, it will be used as a retriever to find the best suited job.

There are two files that require user inputs to run this script. The respective files and their content are explained below:
- `config.yaml` &#8594; Contains the parameters required for the API call to LinkedIn
    - `geoid` &#8594; LinkedIn has a mapping of ids for each country/region. This sequence of numbers will restrict our job scope to a region. The geoid can be updated by performing a manual job search for desired region and extracting the geoid from the url.
    - `field` &#8594; This contains the desired role for the job search
    - `page` &#8594; This defines the number of pages go through when performing the job search

- `skills.yaml` &#8594; Contains the current applicants skills
    - `skills` &#8594; This set of skills will help the retriever idenitfy the best suited job

The output from the code will display a Pandas DataFrame which contains the best suited jobs in descending order as well as the link for a direct job application (if provided from LinkedIn)

### ScrappingDog APIs 
We are using the ScrappingDog free API to LinkedIn to extract the data without being blocked. While the number of credits in the free account is sufficient for a rapid development such as this. For scaling and further usage a premium plan will be required. Due to the limit on the total number of API calls, I have restricted to job extraction to 15 positions during development. For more information on ScrappingDog visit [here](https://www.scrapingdog.com/).

### Ethical Consideration 
While LinkedIn can block the scrapping of private data, job posting fall under the public domain and therefore do not violate the LinkedIn guidelines. The purpose of this project was to implement a naive personalized job search, it is not intended to be used commercially. 