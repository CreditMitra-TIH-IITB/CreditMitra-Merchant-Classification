"""
Indian Names Dataset Generator
===============================
Generates a diverse dataset of Indian person names for the Merchant Classifier.

Coverage:
- 15+ regional/linguistic groups (Hindi, Tamil, Telugu, Bengali, Marathi, etc.)
- Male & Female names
- Multiple religions (Hindu, Muslim, Sikh, Christian, Jain, Parsi)
- Multiple output formats (mimicking UPI payee name variations):
    - "Firstname Lastname"        -> "Ramesh Sharma"
    - "FIRSTNAME LASTNAME"        -> "RAMESH SHARMA"
    - "Firstname"                 -> "Ramesh"
    - "F Lastname"                -> "R Sharma"
    - "Firstname L"               -> "Ramesh S"
    - "firstname lastname"        -> "ramesh sharma"
    - "Firstname Middlename Last" -> "Ramesh Kumar Sharma"

Output: data/indian_person_names.csv
"""

import csv
import random
import os
from itertools import product

# ============================================================================
# FIRST NAMES -- organized by region/language & gender
# ============================================================================

FIRST_NAMES = {
    # ---- NORTH INDIA (Hindi Belt) ----
    "hindi_male": [
        "Ramesh", "Suresh", "Rajesh", "Dinesh", "Mukesh", "Mahesh",
        "Amit", "Sumit", "Rohit", "Mohit", "Vikas", "Deepak",
        "Manoj", "Anil", "Sunil", "Ashok", "Rakesh", "Sanjay",
        "Vivek", "Vinod", "Pramod", "Pankaj", "Ravi", "Ajay",
        "Vijay", "Naveen", "Sachin", "Gaurav", "Arun", "Varun",
        "Nitin", "Lalit", "Manish", "Satish", "Harish", "Girish",
        "Akhil", "Rahul", "Nikhil", "Ankit", "Prashant", "Saurabh",
        "Abhishek", "Shubham", "Ayush", "Kartik", "Harsh", "Yash",
        "Arjun", "Karan", "Ishaan", "Reyansh", "Aarav", "Vihaan",
        "Aditya", "Aryan", "Dhruv", "Kabir", "Ritvik", "Shaurya",
        "Om", "Dev", "Raj", "Krishna", "Shiv", "Ram",
        "Gopal", "Mohan", "Sohan", "Rohan", "Kishan", "Laxman",
        "Bharat", "Chandan", "Hemant", "Jitendra", "Narendra", "Yogesh",
        "Umesh", "Kamal", "Neeraj", "Rajendra", "Surendra", "Devendra",
        # --- expanded ---
        "Abhinav", "Akash", "Alok", "Amresh", "Ankur", "Anshul",
        "Ashish", "Atul", "Balram", "Bhanu", "Brijesh", "Chirag",
        "Dhananjay", "Dharmendra", "Dileep", "Gagan", "Ganesh", "Gaurang",
        "Govind", "Hari", "Hitesh", "Jagdeep", "Jai", "Jayant",
        "Jugal", "Keshav", "Kuldeep", "Kunal", "Lakshay", "Lucky",
        "Madhav", "Manas", "Mayank", "Mrinal", "Naman", "Naresh",
        "Nilesh", "Niraj", "Piyush", "Pratyush", "Puneet", "Pushkar",
        "Raghav", "Rajat", "Raman", "Ranveer", "Rishabh", "Ritesh",
        "Rupesh", "Sandeep", "Sarthak", "Shashank", "Shekhar", "Shivam",
        "Shreyash", "Siddharth", "Sourabh", "Subhash", "Sudhir", "Tarun",
        "Tushar", "Utkarsh", "Vikrant", "Vimal", "Vishal", "Yatin",
        "Bhavesh", "Lokesh", "Mihir", "Anand", "Pranav", "Tanmay",
        "Prakhar", "Aayush", "Darshan", "Divyansh", "Shivansh", "Rudra",
        "Tanuj", "Harshit", "Himanshu", "Ishan", "Jatin", "Kapil",
        "Lovekush", "Mohak", "Nakul", "Omkar", "Parth", "Prateek",
        "Rachit", "Sagar", "Sameer", "Shantanu", "Shrey", "Sparsh",
        "Swapnil", "Tejas", "Ujjwal", "Vedant", "Veer", "Yuvraj",
        "Anurag", "Apoorv", "Ashutosh", "Birendra", "Chandresh", "Durgesh",
        "Gauresh", "Ghanshyam", "Gyanchand", "Harendra", "Jagmohan", "Jeevan",
        "Kailash", "Kamesh", "Kunj", "Lav", "Madhukar", "Nagendra",
        "Navneet", "Nirmal", "Parikshit", "Prabhat", "Ramkumar", "Ranjeet",
        "Satyam", "Shailendra", "Tribhuvan", "Trilok", "Uday", "Vineet",
        "Vipin", "Yashwant", "Ajeet", "Dheeraj", "Indrajeet", "Manmohan",
        "Narayan", "Prem", "Ravindra", "Santosh", "Shyam", "Subodh",
        "Surya", "Triloki", "Vikash", "Yadunath", "Basant", "Devesh",
        "Gyanchand", "Hardik", "Karunesh", "Mithilesh", "Pitambar", "Ratnesh",
        "Shambhu", "Tribhuvan", "Virendra", "Digvijay", "Fateh", "Jagdish",
        "Kunwar", "Lalchand", "Makhan", "Premchand", "Raghunath", "Sarju",
        "Udaybhan", "Chetram", "Dayaram", "Hariom", "Indresh", "Jagram",
        "Lakhan", "Munna", "Nathu", "Pappu", "Ramu", "Shambhu",
        "Tulsi", "Vishnu", "Bholanath", "Deshraj", "Gorelal", "Heeralal",
        "Khemlal", "Motilal", "Pannalal", "Ramswaroop", "Shivnath", "Bansilal",
        "Sitaram", "Ramdayal", "Baldev", "Sukhram", "Phoolchand", "Chaturbhuj",
        "Vishwanath", "Mahadev", "Shankar", "Omprakash", "Ramkishore", "Nandlal",
        "Deendayal", "Dharampal", "Gajanan", "Hukum", "Kanhaiya", "Lakhpat",
        "Madan", "Puran", "Rajaram", "Somnath", "Udayan", "Balwant",
        "Chhotu", "Ghansham", "Indra", "Janardhan", "Keshavrao", "Lekh",
        "Mukund", "Paras", "Rajbir", "Sahadeo", "Trilochan", "Vansh",
    ],
    "hindi_female": [
        "Priya", "Neha", "Pooja", "Anjali", "Sunita", "Anita",
        "Kavita", "Savita", "Rekha", "Meena", "Seema", "Reena",
        "Nisha", "Ritu", "Swati", "Preeti", "Jyoti", "Kiran",
        "Suman", "Asha", "Usha", "Lata", "Geeta", "Sita",
        "Radha", "Mala", "Kamla", "Shobha", "Pushpa", "Sarla",
        "Divya", "Shruti", "Pallavi", "Mansi", "Tanvi", "Kriti",
        "Aditi", "Aarti", "Archana", "Deepika", "Garima", "Isha",
        "Komal", "Monika", "Nikita", "Payal", "Ruchi", "Sakshi",
        "Tanya", "Varsha", "Yukti", "Zara", "Bhavna", "Chhavi",
        "Disha", "Ekta", "Megha", "Shweta", "Sneha", "Sonal",
        "Aanya", "Ananya", "Saanvi", "Myra", "Kiara", "Anika",
        "Pihu", "Navya", "Avni", "Riya", "Tara", "Sia",
        "Khushi", "Muskan", "Simran", "Kajal", "Mamta", "Sapna",
        # --- expanded ---
        "Ankita", "Aparna", "Aradhya", "Bhumi", "Chanchal", "Chandni",
        "Charu", "Damini", "Devika", "Dimple", "Esha", "Falguni",
        "Gauri", "Gunjan", "Harshita", "Heena", "Indu", "Ishika",
        "Janhvi", "Juhi", "Kamini", "Karuna", "Kusum", "Lalita",
        "Madhuri", "Malini", "Manali", "Meenakshi", "Mitali", "Mrinalini",
        "Nandini", "Neelam", "Nimisha", "Padma", "Poonam", "Pratibha",
        "Rachna", "Radhika", "Rashmi", "Renuka", "Richa", "Roshni",
        "Rupali", "Sadhna", "Sangeeta", "Sarita", "Shashi", "Shikha",
        "Shivani", "Srishti", "Sudha", "Surbhi", "Tanu", "Tripti",
        "Vandana", "Vibha", "Vrinda", "Yamini", "Yashika", "Yukta",
        "Aashi", "Alka", "Amrita", "Anamika", "Babita", "Bharti",
        "Bindu", "Chetna", "Daksha", "Durga", "Geetu", "Gita",
        "Hema", "Indra", "Jaya", "Kalpana", "Kanta", "Kusum",
        "Leela", "Mahima", "Maya", "Naina", "Nirmal", "Parvati",
        "Prabha", "Prachi", "Pragati", "Purnima", "Ranjana", "Ratna",
        "Sandhya", "Saroj", "Shakuntala", "Shanti", "Sulochana", "Urmila",
        "Vimla", "Anusha", "Bhawna", "Chitra", "Deepali", "Eksha",
        "Falak", "Gaurvi", "Hansika", "Ishita", "Jannat", "Kashish",
        "Kratika", "Lavanya", "Mishti", "Muskaan", "Nandita", "Ojasvi",
        "Parina", "Priyanshi", "Prerna", "Riddhi", "Siddhi", "Shanaya",
        "Trishala", "Urja", "Vidhi", "Vanya", "Anvi", "Charvi",
        "Dhara", "Eesha", "Himanshi", "Ipsha", "Jagriti", "Koshika",
        "Maira", "Nitya", "Oviya", "Pahi", "Rishika", "Saisha",
        "Tamanna", "Uditi", "Vallari", "Yuvika", "Arushi", "Bhoomi",
        "Chhaya", "Doli", "Guddi", "Hansa", "Indira", "Jamuna",
        "Kaushalya", "Leelawati", "Manorama", "Nirmala", "Phoolmati",
        "Rukmini", "Savitri", "Sumitra", "Tulsi", "Vidyawati", "Yashodha",
        "Gomti", "Halki", "Dhaniya", "Basanti", "Champa", "Durgi",
        "Gulaab", "Hemlata", "Janki", "Kalawati", "Luxmi", "Malti",
        "Panna", "Rambha", "Sheela", "Tejo", "Kammo", "Gudiya",
        "Babli", "Chintu", "Dolly", "Pinky", "Rinku", "Sweety",
        "Lucky", "Lovely", "Mannu", "Nanhi", "Sonu", "Guddo",
    ],

    # ---- SOUTH INDIA (Tamil) ----
    "tamil_male": [
        "Venkatesh", "Ganesh", "Suresh", "Ramesh", "Karthik", "Senthil",
        "Murugan", "Arjun", "Vijay", "Surya", "Siva", "Kumar",
        "Rajan", "Mani", "Selvam", "Bala", "Prakash", "Sundar",
        "Anand", "Bharath", "Dinesh", "Gowtham", "Hari", "Ilango",
        "Jayaram", "Kavin", "Lokesh", "Manikandan", "Naveen", "Prabhu",
        "Rajesh", "Saravanan", "Thirumal", "Udhay", "Vignesh", "Yuvan",
        "Aravind", "Ashwin", "Balaji", "Chandru", "Dhanush", "Elango",
        "Gokul", "Hariharan", "Inbaraj", "Jagan", "Kannan", "Logesh",
        "Mohan", "Nandha", "Parthiban", "Ramkumar", "Shankar", "Tamizh",
        # --- expanded ---
        "Aadhi", "Aakash", "Ajith", "Amarnath", "Anbu", "Arun",
        "Ashokan", "Balamurugan", "Barath", "Boopathi", "Chezhiyan", "Deva",
        "Deepan", "Ezhil", "Gajendran", "Gokulnath", "Gunasekaran", "Guruprasad",
        "Harikrishnan", "Hemachandran", "Imayavaramban", "Iniyan", "Isaivanan",
        "Jagadeesh", "Jayachandran", "Jeeva", "Kalaivanan", "Kalidas", "Kamalakannan",
        "Karuppiah", "Kathiresan", "Kaviarasan", "Kiran", "Kumaresan", "Kumaravel",
        "Lakshmanan", "Loganathan", "Madhan", "Mahendran", "Malaichamy", "Manigandan",
        "Marimuthu", "Meganathan", "Mohanraj", "Moorthy", "Muthukumar", "Muthuraj",
        "Nagaraj", "Nallasivam", "Natarajan", "Nedunchezhiyan", "Nithyanandan",
        "Padmanaban", "Palanivel", "Pandian", "Paramasivam", "Paventhan",
        "Perumal", "Ponraj", "Prabakaran", "Prabakar", "Pradeep", "Premkumar",
        "Raghunath", "Rajasekar", "Rajavel", "Ramachandran", "Ramanathan", "Ramprasad",
        "Rathinavelu", "Ravichandran", "Ravikumar", "Sakthivel", "Samuvel", "Sankar",
        "Santhosh", "Sarath", "Sasikumar", "Sathish", "Sathyamoorthy", "Selvakumar",
        "Senthilkumar", "Sethuraman", "Shanmugasundaram", "Silambarasan", "Sivakumar",
        "Sivanesan", "Somasundaram", "Sridhar", "Sriram", "Subramani", "Sundaresan",
        "Surendran", "Thamilarasan", "Thangadurai", "Thangavel", "Thirunavukkarasu",
        "Udayakumar", "Vaithiyanathan", "Vasanthan", "Veerapandi", "Velmurugan",
        "Venkatachalam", "Venugopal", "Vetriselvan", "Vijayakumar", "Vijayaraj",
        "Vinayagam", "Vinoth", "Yoganathan", "Yuvaraj", "Kalaiselvan",
        "Azhagappan", "Chelladurai", "Devaraj", "Dharmaraj", "Ekambaram",
        "Ganapathi", "Govindan", "Ilaiyaraja", "Jeyapaul", "Kandasamy",
        "Mahalingam", "Narayanan", "Palani", "Rajagopal", "Sampath",
        "Thalapathi", "Vasudevan", "Annamalai", "Chinnasamy", "Duraipandi",
        "Ganesamoorthy", "Iyyappan", "Jayaraman", "Kothandaraman", "Lakshmipathy",
        "Manoharan", "Namasivayam", "Palanisamy", "Ranganathan", "Subbian",
        "Thiagarajan", "Vaidyanathan", "Arivazhagan", "Balaganesh", "Chandramohan",
        "Devendran", "Ezhumalai", "Gnanasambandam", "Ilayaperumal", "Jeganathan",
        "Kuppusamy", "Lakshmikanthan", "Mahadev", "Nataraj", "Palaniappan",
    ],
    "tamil_female": [
        "Lakshmi", "Saroja", "Meena", "Revathi", "Priya", "Divya",
        "Kavitha", "Sangeetha", "Deepa", "Nithya", "Padma", "Vani",
        "Janani", "Keerthana", "Lavanya", "Madhumitha", "Nandhini",
        "Oviya", "Pavithra", "Ranjani", "Sowmya", "Thenmozhi",
        "Uma", "Vasanthi", "Yamuna", "Abinaya", "Bhuvana", "Chitra",
        "Dharani", "Eswari", "Gayathri", "Hema", "Indira", "Kalpana",
        "Latha", "Mala", "Nalini", "Parvathi", "Radha", "Saradha",
        "Anusha", "Aishwarya", "Dhivya", "Harini", "Iswarya", "Jothika",
        # --- expanded ---
        "Amudha", "Anbukkarasi", "Anitha", "Archana", "Arulmozhi",
        "Bhuvaneshwari", "Brinda", "Chellammal", "Chellam", "Devi",
        "Dhanalakshmi", "Gomathi", "Gowri", "Hemalatha", "Hemavathi",
        "Ilakiya", "Indumathi", "Iswaryalakshmi", "Jayalakshmi", "Jayanthi",
        "Jeyashree", "Kala", "Kalaiselvi", "Kamala", "Kanagavalli",
        "Kanmani", "Kanimozhi", "Kannamma", "Karunambigai", "Kasthuri",
        "Kiruthika", "Kokilavani", "Kumari", "Kuppamal", "Lalitha",
        "Leelavathi", "Mahalakshmi", "Maheswari", "Mallika", "Manimegalai",
        "Mangai", "Mangalam", "Meenakshi", "Menaka", "Mohana",
        "Muthulakshmi", "Nagalakshmi", "Nagammal", "Nirmala", "Nivedha",
        "Parimala", "Ponni", "Poornima", "Pushpalatha", "Rajalakshmi",
        "Rajeshwari", "Rajeswari", "Ramya", "Renuga", "Rohini",
        "Rukmani", "Saranya", "Sarojini", "Sasikala", "Savithri",
        "Selvi", "Shanthi", "Sivakami", "Sornalatha", "Suganya",
        "Sumathi", "Sundari", "Surya", "Tamilarasi", "Tamilselvi",
        "Thangam", "Thilagavathi", "Thirumagal", "Umamaheswari", "Valarmathi",
        "Varalakshmi", "Vasuki", "Vedavalli", "Velankanni", "Vijayalakshmi",
        "Ambika", "Annapoorani", "Bharathi", "Bhavani", "Chandrika",
        "Devapriya", "Durgadevi", "Elavarasi", "Gnanavalli", "Gowthami",
        "Iyothee", "Jagathambal", "Jeyarani", "Kamatchi", "Kaveri",
        "Lakshmipriya", "Manonmani", "Marudhammal", "Meenachi", "Murugeshwari",
        "Nachiyar", "Nagaveni", "Neelaveni", "Pachiammal", "Pappammal",
        "Periyammal", "Ponnammal", "Radhika", "Rajammal", "Ranganayaki",
        "Sakunthala", "Seetha", "Senthamarai", "Sivagami", "Subbalakshmi",
        "Subbulakshmi", "Thaiyalnayagi", "Valliammai", "Veeralakshmi",
    ],

    # ---- SOUTH INDIA (Telugu) ----
    "telugu_male": [
        "Venkat", "Srikanth", "Sudheer", "Mahesh", "Ravi", "Srinivas",
        "Narasimha", "Prasad", "Satish", "Ramana", "Kishore", "Rajendra",
        "Anil", "Bhaskar", "Chaitanya", "Durga", "Eswar", "Gopi",
        "Harsha", "Jagadish", "Kalyan", "Laxminarayana", "Madhu", "Nagendra",
        "Pavan", "Raghu", "Sai", "Tarun", "Uday", "Varun",
        "Ajay", "Bhanu", "Chiranjeevi", "Deepak", "Ganesh", "Hari",
        "Kiran", "Mohan", "Naresh", "Phani", "Raju", "Sunil",
        # --- expanded ---
        "Amarendra", "Anjaneyulu", "Apparao", "Ashok", "Balaraju", "Balakrishna",
        "Bhagavan", "Brahmanandam", "Chakrapani", "Chandrasekhar", "Damodar",
        "Dasarath", "Devadas", "Dharmarao", "Govardhan", "Hanumantha",
        "Haribabu", "Hemachandra", "Jaganmohan", "Janardhan", "Jayaprakash",
        "Kalidasu", "Kameswara", "Kesava", "Kodandarami", "Kondal",
        "Lakshman", "Lingam", "Madhusudan", "Mallikarjun", "Manohar",
        "Murali", "Nagababu", "Nagarjuna", "Nageswara", "Nandakishore",
        "Narsaiah", "Padmanabham", "Peddiraju", "Penchalaiah", "Pitchaiah",
        "Purushottam", "Radhakrishna", "Raghuram", "Rajasekhar", "Ramakrishna",
        "Ramalingam", "Ramamurthy", "Ramprasad", "Rangaiah", "Ravindra",
        "Sambasiva", "Sanjeev", "Sathyanarayana", "Seshadri", "Shankar",
        "Sivaprasad", "Someswara", "Sreekanth", "Sreerama", "Subbarao",
        "Sudhakar", "Suryaprakash", "Tirumala", "Tulasidas", "Umamaheshwara",
        "Veerabhadra", "Venkataiah", "Venumadhav", "Viswanath", "Yellaiah",
        "Adinarayana", "Bapuji", "Chandrababu", "Damodaram", "Eknath",
        "Ganapathi", "Harinath", "Indrasena", "Janakidam", "Kailasam",
        "Lakshmikant", "Madhusudhan", "Nagamani", "Obul", "Parasuram",
        "Raghavendra", "Satyam", "Tirupathi", "Upendra", "Vamsi",
        "Vinay", "Yaswanth", "Ajit", "Bharadwaj", "Charan",
        "Dinkar", "Gautham", "Hemanth", "Jeevan", "Karthikeya",
        "Lokesh", "Manideep", "Nihal", "Prem", "Rohith",
        "Saikiran", "Teja", "Vishal", "Aakash", "Bunny",
        "Dinakar", "Goutham", "Himanshu", "Jyothirmayi", "Kaushik",
        "Manikanta", "Nikhilesh", "Praneeth", "Rakshith", "Siddharth",
        "Trilok", "Vikranth", "Abhinav", "Chandra", "Dheeraj",
    ],
    "telugu_female": [
        "Sravani", "Bhavani", "Lakshmi", "Padmavathi", "Swathi", "Anusha",
        "Divya", "Mounika", "Spandana", "Tejaswini", "Harika", "Sushma",
        "Akhila", "Bindu", "Chandana", "Deepthi", "Geetha", "Hymavathi",
        "Indumathi", "Jhansi", "Kalyani", "Madhavi", "Naga", "Pallavi",
        "Rajeshwari", "Saritha", "Tulasi", "Usha", "Vijaya", "Yamini",
        # --- expanded ---
        "Amrutha", "Anitha", "Aruna", "Asha", "Bharathi", "Bhargavi",
        "Chamundeshwari", "Chiranjeevi", "Damayanthi", "Deepavali", "Durga",
        "Eswaramma", "Ganga", "Gayathri", "Girija", "Hemalatha",
        "Jaya", "Jayalalitha", "Kamala", "Kanaka", "Kasthuri",
        "Kranthi", "Kusuma", "Lalitha", "Laxmidevi", "Leela",
        "Madhulatha", "Mahalakshmi", "Manga", "Meenakshi", "Nagamani",
        "Nagalakshmi", "Nirmala", "Padma", "Parvathi", "Prameela",
        "Prashanthi", "Pushpavathi", "Radha", "Radhika", "Ramadevi",
        "Rani", "Revathi", "Rohini", "Sailaja", "Sandhya",
        "Saraswathi", "Seetha", "Sharada", "Shobha", "Sita",
        "Sitamahalakshmi", "Sowjanya", "Srilatha", "Subhadra", "Sucharitha",
        "Suguna", "Sulochana", "Sumalatha", "Sunitha", "Supriya",
        "Surekha", "Swaroopa", "Triveni", "Uma", "Vanaja",
        "Vasantha", "Varalakshmi", "Veena", "Vijayalakshmi", "Vimala",
        "Vinoda", "Anuradha", "Aparna", "Archana", "Anjali",
        "Dhanalakshmi", "Krishnaveni", "Manga", "Nagarathnamma", "Prabhavathi",
        "Ramulamma", "Sarojini", "Vaidehi", "Vasundara", "Vydehi",
        "Ahalya", "Bhagyalakshmi", "Chandra", "Damayanti", "Eswari",
        "Gnana", "Hemavathi", "Indira", "Janaki", "Koteswaramma",
    ],

    # ---- SOUTH INDIA (Kannada) ----
    "kannada_male": [
        "Suresh", "Ramesh", "Prashanth", "Manjunath", "Basavaraj",
        "Shivaraj", "Girish", "Harish", "Naveen", "Praveen",
        "Arun", "Chetan", "Darshan", "Ganesh", "Keshav",
        "Lokesh", "Madhu", "Nandan", "Pavan", "Raghavendra",
        "Sachin", "Tejas", "Varun", "Yashwanth", "Abhishek",
        # --- expanded ---
        "Ajith", "Akshay", "Amarnath", "Anand", "Anirudh", "Appaji",
        "Ashoka", "Balagangadhar", "Bharath", "Channappa", "Channabasappa",
        "Dasappa", "Devaraja", "Dharmendra", "Dinakar", "Ganapathi",
        "Gurappa", "Guru", "Hanumesh", "Jagadish", "Jayanna",
        "Jayaprakash", "Karthik", "Kempegowda", "Krishnappa", "Kumara",
        "Lakshmipathi", "Lingaraju", "Mahadev", "Mahadeva", "Mallikarjun",
        "Manju", "Mariswamy", "Mruthyunjaya", "Muniswamy", "Murthy",
        "Nagappa", "Nagaraj", "Nanjunda", "Nikhil", "Ninga",
        "Omkar", "Paramesh", "Prakash", "Puttaswamy", "Raghunandan",
        "Rajanna", "Rajkumar", "Ramesha", "Rangaswamy", "Ravikumar",
        "Rudresh", "Sadashiva", "Santhosh", "Sathish", "Shankar",
        "Shivakumar", "Shivanna", "Shrinivas", "Siddalingappa", "Siddappa",
        "Srikanta", "Subbanna", "Subbiah", "Sudhakar", "Thimmappa",
        "Umesh", "Veeranna", "Veerappa", "Venkatappa", "Venkatesh",
        "Vijaykumar", "Vinay", "Vishwanath", "Yellappa", "Yogesh",
        "Anantha", "Basappa", "Chamaraj", "Devaiah", "Eranna",
        "Gavisiddappa", "Hanumanthappa", "Ishwar", "Jayaraj", "Kalappa",
        "Lingappa", "Mahabaleshwar", "Nagaraja", "Onkarappa", "Papanna",
        "Rajashekar", "Sampathkumar", "Tippeswamy", "Udaykumar", "Virupaksha",
    ],
    "kannada_female": [
        "Asha", "Bhagya", "Chaitra", "Deepa", "Gowri",
        "Jayashree", "Kavya", "Laxmi", "Meghana", "Nandini",
        "Poornima", "Rashmi", "Shilpa", "Tanuja", "Vidya",
        "Akshata", "Brinda", "Chandana", "Divya", "Harshitha",
        # --- expanded ---
        "Aishu", "Ambika", "Amulya", "Anitha", "Anuradha",
        "Arathi", "Ashwini", "Bhavya", "Champa", "Chanamma",
        "Deeksha", "Devamma", "Dhanya", "Geetha", "Girija",
        "Hemalatha", "Indumathi", "Jyothi", "Kala", "Kamala",
        "Kasthuri", "Keerthana", "Kusuma", "Lalitha", "Lakshmi",
        "Malathi", "Mangala", "Manjula", "Meena", "Nagamma",
        "Nagarathna", "Nalini", "Nirmala", "Padma", "Parvathi",
        "Pavithra", "Prabhavathi", "Priyanka", "Pushpa", "Radha",
        "Radhika", "Rajeshwari", "Ranjitha", "Renuka", "Roopa",
        "Rukmini", "Savitha", "Shantha", "Sharada", "Shobha",
        "Sowmya", "Srilakshmi", "Sudha", "Suma", "Sumathi",
        "Sunitha", "Supreetha", "Swapna", "Triveni", "Uma",
        "Usha", "Vasanthi", "Veda", "Vijayalakshmi", "Yashodha",
        "Annapurna", "Bhairavi", "Chinnamma", "Durgamma", "Gangamma",
        "Hanumamma", "Ishwaramma", "Jayamma", "Kamalamma", "Lingamma",
        "Mahadevi", "Nagalakshmi", "Obavva", "Puttamma", "Rajamma",
        "Sharadamma", "Thimmamma", "Umadevi", "Veeramma", "Yellamma",
    ],

    # ---- SOUTH INDIA (Malayalam) ----
    "malayalam_male": [
        "Arun", "Biju", "Dileep", "Gopan", "Hari", "Jayesh",
        "Kiran", "Manoj", "Nishad", "Pramod", "Rajesh", "Sajeev",
        "Sunil", "Unni", "Vishnu", "Akhil", "Bipin", "Deepu",
        "Gibin", "Jibin", "Lijin", "Midhun", "Nikhil", "Riyas",
        "Shibu", "Vipin", "Anoop", "Jithin", "Pradeep", "Sreejith",
        # --- expanded ---
        "Abin", "Ajeesh", "Ajin", "Aneesh", "Anil", "Anish",
        "Anto", "Arjun", "Arun", "Ashok", "Babu", "Baiju",
        "Bijoy", "Binoy", "Blesson", "Bobby", "Boby", "Chandran",
        "Dasan", "Deepak", "Dileep", "Eldhose", "Eldho", "Geo",
        "Gireesh", "Gopakumar", "Gopinath", "Harikrishnan", "Jayan",
        "Jayaraj", "Jayesh", "Jeevan", "Jijin", "Jijo", "Jobin",
        "Joby", "John", "Jomon", "Joshy", "Justin", "Kannan",
        "Krishnakumar", "Kuttan", "Laiju", "Lijo", "Madhu", "Mahesh",
        "Manu", "Martin", "Mathew", "Mohan", "Mohanan", "Murali",
        "Narayanan", "Nishant", "Noufal", "Prabhakaran", "Prakash",
        "Pramod", "Prasanth", "Radhakrishnan", "Rajeev", "Rajan",
        "Rajeevan", "Raju", "Ramachandran", "Raveendran", "Robin",
        "Sabu", "Sajith", "Sajan", "Salu", "Sanal", "Santhosh",
        "Sarathkumar", "Shaji", "Shameer", "Shari", "Shiju", "Sijin",
        "Siju", "Sivadas", "Sreekumar", "Sreekanth", "Sreenivasan",
        "Sudheer", "Sumod", "Suresh", "Thomas", "Tinu", "Tomy",
        "Unnikrishnan", "Venu", "Vijayan", "Viju", "Vinay", "Vinod",
        "Vishwanath", "Yohannan", "Zachariah", "Abdul", "Afsal", "Ajmal",
        "Ashraf", "Faizal", "Firoz", "Hameed", "Jabir", "Jaseem",
        "Kunju", "Najeeb", "Rasheed", "Salim", "Shamsu", "Shafeeq",
        "Siraj", "Suhail", "Ubaid", "Ummer", "Wahab", "Yahya",
    ],
    "malayalam_female": [
        "Anu", "Bindu", "Deepa", "Geetha", "Jaya", "Kavitha",
        "Lekha", "Mini", "Neethu", "Priya", "Reshma", "Suja",
        "Veena", "Ammu", "Devika", "Gayathri", "Lakshmi", "Meera",
        "Parvathy", "Revathy", "Swapna", "Athira", "Gopika", "Keerthi",
        # --- expanded ---
        "Aiswarya", "Ajitha", "Amala", "Amina", "Ancy", "Anila",
        "Anitha", "Anju", "Anna", "Aparna", "Archana", "Arya",
        "Beena", "Bincy", "Bijimol", "Chinju", "Chithra", "Devi",
        "Dhanya", "Divya", "Elsy", "Fathima", "Gracy", "Haseena",
        "Indhu", "Indu", "Jancy", "Jaseela", "Jayasree", "Jisha",
        "Jisha", "Jolly", "Josephine", "Kala", "Kalyani", "Kamala",
        "Karthika", "Latha", "Lily", "Lisha", "Lissy", "Manju",
        "Meeraba", "Mercy", "Minimol", "Moli", "Nandana", "Nisha",
        "Pankaja", "Ponnamma", "Pushpa", "Radhika", "Rajani", "Rajitha",
        "Rema", "Renjini", "Rini", "Roshni", "Saleena", "Saraswathi",
        "Saritha", "Sathya", "Seema", "Shailaja", "Shajna", "Sheeba",
        "Sheena", "Sherly", "Shiji", "Sini", "Smitha", "Sobha",
        "Sonia", "Sreekala", "Sreelatha", "Sudha", "Sunitha", "Supriya",
        "Tessy", "Thankam", "Thulasi", "Usha", "Vasantha", "Vijaya",
        "Vineetha", "Yamuna", "Zainaba", "Beevi", "Jameela", "Khadeeja",
        "Mariyam", "Nabeesa", "Noorjahan", "Pathumma", "Rahmathunnisa",
        "Safiya", "Sainaba", "Subaida", "Suhara", "Wahida", "Zubaida",
    ],

    # ---- EAST INDIA (Bengali) ----
    "bengali_male": [
        "Subhash", "Debashis", "Partha", "Arnab", "Sourav", "Aniket",
        "Dipankar", "Kaushik", "Rana", "Sayan", "Arijit", "Bikash",
        "Chiranjit", "Debjit", "Gautam", "Himanshu", "Jayanta", "Koushik",
        "Mrinmoy", "Niladri", "Prosenjit", "Rajat", "Saikat", "Tanmoy",
        "Ujjwal", "Abhijit", "Biswajit", "Debojyoti", "Indranil", "Soumyajit",
        "Arka", "Ritwick", "Anirban", "Sabyasachi", "Subrata", "Tapas",
        # --- expanded ---
        "Abir", "Adrish", "Agnimitra", "Akash", "Alok", "Amal",
        "Amitava", "Ananda", "Aniruddha", "Anup", "Apurba", "Arindam",
        "Aritra", "Arpan", "Ashim", "Asit", "Atin", "Avijit",
        "Baidyanath", "Bankim", "Basudeb", "Bibhuti", "Bidhan", "Bijoy",
        "Biplab", "Biswarup", "Buddhadeb", "Chandranath", "Chitta",
        "Debabrata", "Debasis", "Debkumar", "Debu", "Dhiman", "Dilip",
        "Dipak", "Gagan", "Goutam", "Hiran", "Hrishikesh", "Jiban",
        "Jishnu", "Joydeep", "Kallol", "Kanchan", "Kartik", "Keshab",
        "Krishnendu", "Kunal", "Mahadeb", "Malay", "Manab", "Manas",
        "Manik", "Mihir", "Monojit", "Mrinal", "Nabakumar", "Nabin",
        "Naren", "Narayan", "Nilanjan", "Nripen", "Pallab", "Paritosh",
        "Parthasarathi", "Pijush", "Pinaki", "Prabir", "Pradip", "Pralay",
        "Pranab", "Prasanta", "Prashanta", "Pratul", "Probir", "Probal",
        "Prodip", "Promod", "Pulak", "Purnendu", "Rabindranath", "Raju",
        "Ramkrishna", "Ranajit", "Ranjit", "Rathin", "Rupak", "Samar",
        "Sambhu", "Sandip", "Sanjib", "Santanu", "Sarit", "Saumitra",
        "Shankhadeep", "Shantimoy", "Shirshendu", "Shyamal", "Siddharta",
        "Sisir", "Snehashish", "Soumen", "Soumik", "Souvik", "Sovan",
        "Subhankar", "Subir", "Subroto", "Sudeshna", "Sudipta", "Sukanta",
        "Sukumar", "Sumantra", "Sumit", "Sunirmal", "Supriyo", "Surojit",
        "Susanta", "Sushanta", "Tapan", "Tarun", "Tushar", "Utpal",
        "Uttam", "Bikram", "Chittaranjan", "Dhiraj", "Gobinda", "Himangshu",
        "Joydeb", "Kalipada", "Madhusudan", "Nirmal", "Priyabrata",
        "Raghunath", "Satyen", "Tarak", "Upendranath", "Bireswar",
    ],
    "bengali_female": [
        "Arpita", "Banhi", "Chandrima", "Debjani", "Gargi", "Ipsita",
        "Jayeeta", "Keya", "Laboni", "Moumita", "Nandita", "Paramita",
        "Rituparna", "Satabdi", "Tanushree", "Aditi", "Anindita", "Bidisha",
        "Deboleena", "Ishita", "Kamalika", "Mitali", "Payel", "Reeya",
        "Sayantani", "Trisha", "Swastika", "Raima", "Paoli", "Mimi",
        # --- expanded ---
        "Adrija", "Ahana", "Aindrila", "Aishani", "Amrapali", "Ananya",
        "Angana", "Anindita", "Ankita", "Apala", "Aparajita", "Arundhati",
        "Basabi", "Bharati", "Bithika", "Bonani", "Bristi", "Chitrangada",
        "Chumki", "Damayanti", "Debika", "Deepannita", "Dipannita",
        "Durba", "Ekani", "Gandhari", "Indrani", "Jaba", "Jharna",
        "Jhilmil", "Jolly", "Jugandhara", "Kalyani", "Kamala", "Kanan",
        "Kankabati", "Kaushiki", "Keka", "Ketaki", "Khela", "Koneenica",
        "Leena", "Lopa", "Lopamudra", "Madhumita", "Mahua", "Mallika",
        "Mamata", "Mandira", "Manisha", "Meghna", "Mekhala", "Minakshi",
        "Mohua", "Monideepa", "Monisha", "Mouli", "Mousumi", "Mukta",
        "Nabaneeta", "Nabanita", "Nilanjana", "Nirmala", "Nishita",
        "Pallabi", "Papiya", "Piyali", "Pompa", "Poulomi", "Prarthana",
        "Pratima", "Purnima", "Pushpita", "Raka", "Reshmi", "Rima",
        "Rimi", "Rimjhim", "Rinki", "Riti", "Roopa", "Rupa",
        "Rupsha", "Sabita", "Saheli", "Sankari", "Saptami", "Sarani",
        "Sarasi", "Shaoli", "Sharmistha", "Shinjini", "Shrabani", "Shreya",
        "Shubhra", "Sohini", "Soma", "Sonali", "Subarna", "Suchorita",
        "Sudipta", "Sugata", "Sumita", "Supriti", "Susmita", "Sutapa",
        "Tania", "Tithi", "Trishna", "Tumpa", "Ujjayini", "Urmi",
        "Ushashi", "Anushka", "Deepshikha", "Jhilik", "Koeli", "Lakshmi",
        "Sohagi", "Madhabi", "Nivedita", "Phuleshwari", "Savitri",
    ],

    # ---- EAST INDIA (Odia) ----
    "odia_male": [
        "Bikash", "Chitta", "Debendra", "Gagan", "Hemanta", "Jitendra",
        "Kishore", "Lingaraj", "Manoj", "Niranjan", "Pradeep", "Rabindra",
        "Sarat", "Tapan", "Umakanta", "Ajit", "Brundaban", "Dharani",
        # --- expanded ---
        "Abhimanyu", "Achyut", "Akshaya", "Amiya", "Ananta", "Anil",
        "Ashutosh", "Baidyanath", "Banamali", "Basanta", "Bhagaban",
        "Bibekananda", "Bidyadhar", "Bijayananda", "Bipin", "Biranchi",
        "Bishnu", "Byomakesh", "Chaitanya", "Chakradhar", "Chandra",
        "Damodar", "Deba", "Dhirendra", "Dibakar", "Dillip",
        "Durgacharan", "Fakir", "Ganeswar", "Ghanashyam", "Gopabandhu",
        "Gopinath", "Gourahari", "Harihar", "Jagabandhu", "Jagannath",
        "Jalandhar", "Jaydev", "Kailash", "Kanhei", "Kartik",
        "Kedarnath", "Keshab", "Khirod", "Krushna", "Kulamani",
        "Laxmidhar", "Lokanath", "Madhusudan", "Maheswar", "Manoranjan",
        "Muralidhar", "Nabakishore", "Narahari", "Narayan", "Narasingha",
        "Nilamani", "Nilamadhab", "Niranjan", "Padmanabh", "Paramananda",
        "Prafulla", "Prakash", "Prasanna", "Purna", "Purushottam",
        "Raghunath", "Rajkishore", "Ramakanta", "Ramchandra", "Ratan",
        "Sadananda", "Sahadev", "Sarbeswar", "Shyamsundar", "Somanath",
        "Subash", "Sushant", "Tribikram", "Trinath", "Udayanath",
    ],
    "odia_female": [
        "Anuradha", "Basanti", "Chandrama", "Draupadi", "Gouri",
        "Harapriya", "Iti", "Jasmine", "Kabita", "Lipsa",
        "Mamata", "Nibedita", "Puja", "Ranjita", "Sasmita",
        # --- expanded ---
        "Ambika", "Anita", "Anjana", "Aparna", "Archana",
        "Arunima", "Bandita", "Barsha", "Bhagyalata", "Bijayini",
        "Bijaylaxmi", "Bina", "Charulata", "Chinmayi", "Debajani",
        "Deepanjali", "Eti", "Gita", "Gitanjali", "Hiranmayi",
        "Ilina", "Janaki", "Jyotirmayee", "Kalpana", "Kamini",
        "Kanchan", "Kasturi", "Kiranbala", "Laxmipriya", "Lila",
        "Lopamudra", "Madhabi", "Malaya", "Manasi", "Minakshi",
        "Mita", "Monalisha", "Nalini", "Nirupama", "Niyati",
        "Padmalaya", "Pallabi", "Pramila", "Pratibha", "Priyambada",
        "Purnima", "Pushpalata", "Rajalaxmi", "Rashmita", "Renubala",
        "Rupali", "Sabita", "Sandhyarani", "Sanjukta", "Sarojini",
        "Shakuntala", "Shanti", "Sita", "Smita", "Snigdha",
        "Subhadra", "Sukanti", "Sulochana", "Sunanda", "Supriya",
        "Sushree", "Tapasya", "Trupti", "Urmila", "Usha",
    ],

    # ---- WEST INDIA (Marathi) ----
    "marathi_male": [
        "Sachin", "Ganesh", "Sunil", "Anil", "Ajay", "Sagar",
        "Amol", "Nilesh", "Sandip", "Pravin", "Swapnil", "Tushar",
        "Yogesh", "Atul", "Bhushan", "Chetan", "Datta", "Hemant",
        "Kiran", "Laxman", "Milind", "Ninad", "Omkar", "Prasad",
        "Rahul", "Santosh", "Tanmay", "Vaibhav", "Yashwant", "Akshay",
        "Abhijeet", "Rohan", "Shripad", "Vitthal", "Dnyaneshwar", "Balasaheb",
        # --- expanded ---
        "Adinath", "Ajinkya", "Ameya", "Amey", "Ankush", "Avinash",
        "Babasaheb", "Baban", "Bajirao", "Bhalachandra", "Bhausaheb",
        "Chandrakant", "Chandrashekhar", "Dagadu", "Dashrath", "Dattatray",
        "Devdatta", "Dhanaji", "Dhananjay", "Digambar", "Dilip",
        "Eknath", "Ganpat", "Gokul", "Gopal", "Govind", "Hanumant",
        "Harishchandra", "Jagdish", "Jayant", "Kaustubh", "Keshav",
        "Kishor", "Kondiba", "Krishna", "Madhav", "Mahadev",
        "Mahendra", "Malhar", "Mangesh", "Maruti", "Moreshwar",
        "Mukund", "Nagnath", "Namdev", "Nana", "Narayan",
        "Narhar", "Nikhil", "Pandurang", "Prabhakar", "Pralhad",
        "Pundlik", "Purushottam", "Raghunath", "Rajaram", "Ramchandra",
        "Ramdas", "Ratan", "Ravindra", "Sadashiv", "Sambhaji",
        "Sanjog", "Shankar", "Shantaram", "Sharad", "Shivaji",
        "Shridhar", "Shrikanth", "Siddharth", "Somnath", "Subhash",
        "Sudhakar", "Suhas", "Suresh", "Tanaji", "Tatya",
        "Trimbak", "Tukaram", "Uddhav", "Umesh", "Vasant",
        "Vinayak", "Viraj", "Vishwas", "Yatin", "Rajesh",
        "Ashok", "Chandrakant", "Devendra", "Ganpatrao", "Hari",
        "Jaydeep", "Kedar", "Madhukar", "Nandkishor", "Onkar",
        "Pandit", "Ramkrishna", "Sadanand", "Tryambak", "Uttam",
        "Bharat", "Dadasaheb", "Gajanan", "Jagannath", "Krushna",
        "Limbaji", "Manikrao", "Nanasaheb", "Pratap", "Raosaheb",
        "Shankarrao", "Vinod", "Anant", "Balkrishna", "Chatrapati",
    ],
    "marathi_female": [
        "Aarti", "Bhagyashree", "Chaitali", "Deepali", "Gauri",
        "Hemangi", "Isha", "Jayashri", "Ketaki", "Leena",
        "Manisha", "Nandini", "Pallavi", "Rujuta", "Sarika",
        "Tejal", "Ujwala", "Vrushali", "Ashwini", "Smita",
        "Snehal", "Prajakta", "Madhura", "Harshada", "Sharvari",
        # --- expanded ---
        "Aishwarya", "Akanksha", "Amruta", "Anagha", "Anjali",
        "Anuja", "Aparna", "Archana", "Asmita", "Avanti",
        "Bhairavi", "Bhavana", "Charushila", "Chitra", "Daksha",
        "Dhanashree", "Dipali", "Durga", "Gandhari", "Geetanjali",
        "Girija", "Hema", "Indira", "Janhavi", "Jayanti",
        "Jyotsna", "Kanchan", "Karishma", "Kaveri", "Kiran",
        "Komal", "Lata", "Laxmibai", "Madhavi", "Manda",
        "Mangal", "Meena", "Meenal", "Meera", "Mohini",
        "Mrinal", "Mukta", "Naina", "Nirmala", "Padma",
        "Pramila", "Pranali", "Pranjali", "Pratiksha", "Prerana",
        "Priti", "Purva", "Rachana", "Radhika", "Rajani",
        "Rani", "Renuka", "Rohini", "Rukmini", "Rupali",
        "Sakhi", "Sandhya", "Sanjivani", "Sarojini", "Savita",
        "Shaila", "Shailaja", "Shakuntala", "Shamika", "Shantabai",
        "Sharda", "Shital", "Shobha", "Shubhada", "Suhasini",
        "Sulabha", "Sunanda", "Sunetra", "Supriya", "Suvarna",
        "Swara", "Swati", "Tara", "Vaishali", "Vandana",
        "Varsha", "Vasudha", "Vidya", "Vijaya", "Vimal",
        "Vinaya", "Yamuna", "Yashodha", "Yogita", "Chhaya",
        "Indumati", "Janabai", "Kamalbai", "Laxmibai", "Muktabai",
        "Parvati", "Radhabai", "Sakhubai", "Vithabai", "Ahilya",
    ],

    # ---- WEST INDIA (Gujarati) ----
    "gujarati_male": [
        "Jayesh", "Ketan", "Nilesh", "Paresh", "Rakesh", "Sanjay",
        "Tejas", "Uday", "Vipul", "Ashwin", "Bhavin", "Chirag",
        "Darshan", "Gaurang", "Hardik", "Jignesh", "Kamlesh", "Mitesh",
        "Nimesh", "Parth", "Rajan", "Siddharth", "Tushar", "Yash",
        "Dharmesh", "Hiren", "Jigar", "Kunal", "Maulik", "Nishant",
        # --- expanded ---
        "Aarav", "Ajit", "Alpesh", "Amish", "Ankit", "Ankur",
        "Archit", "Arpit", "Atit", "Atman", "Bakul", "Bharat",
        "Bhaskar", "Bhavesh", "Bhupendra", "Birju", "Chaitanya", "Chandresh",
        "Chintan", "Darpan", "Daval", "Deven", "Devang", "Dhaval",
        "Dhiren", "Dilip", "Dipak", "Divyesh", "Falgun", "Gautam",
        "Girish", "Gopi", "Gunvant", "Harshal", "Hemang", "Hitesh",
        "Indravadan", "Ishwar", "Jagdish", "Jalpa", "Janmesh", "Jatin",
        "Jeet", "Jinal", "Jinendra", "Kalpesh", "Kanaiya", "Kandarp",
        "Kaushal", "Keval", "Keyur", "Kinjal", "Kirti", "Kunj",
        "Lalit", "Lavesh", "Mahendra", "Manhar", "Manish", "Mitul",
        "Mohan", "Mukesh", "Naimish", "Nalin", "Nand", "Narendra",
        "Naval", "Nayan", "Niket", "Nikunj", "Nirav", "Nirbhay",
        "Ojas", "Omesh", "Pankaj", "Parin", "Paritosh", "Piyush",
        "Pragnesh", "Prakash", "Pranav", "Pratik", "Pravin", "Pritesh",
        "Puneet", "Pushpak", "Raghu", "Rajiv", "Ramesh", "Rushil",
        "Sachin", "Sagar", "Samir", "Sandeep", "Sanjiv", "Satish",
        "Shailesh", "Shalin", "Sharad", "Shreyas", "Sudhir", "Sumant",
        "Sunil", "Suresh", "Tarak", "Utsav", "Vasant", "Vatsal",
        "Vijay", "Vilas", "Vinay", "Viral", "Vishal", "Vivek",
    ],
    "gujarati_female": [
        "Bhavna", "Darshana", "Hetal", "Jigna", "Krupa", "Minal",
        "Nidhi", "Payal", "Rinal", "Sweta", "Tejal", "Urvi",
        "Vaishali", "Ankita", "Foram", "Heena", "Janki", "Khyati",
        "Manali", "Nehal", "Prachi", "Riddhi", "Siddhi", "Twinkle",
        # --- expanded ---
        "Aarti", "Alpa", "Amisha", "Anjali", "Anshi", "Anuradha",
        "Archana", "Bhumi", "Bina", "Champa", "Chandni", "Charmi",
        "Chetna", "Daxini", "Deepa", "Dhara", "Dhruvi", "Dimple",
        "Disha", "Divya", "Drashti", "Ekta", "Falak", "Feni",
        "Gaurvi", "Gayatri", "Gita", "Harsha", "Hina", "Ila",
        "Ilaben", "Indira", "Isha", "Jagruti", "Jasmin", "Jaya",
        "Jinal", "Jyoti", "Kajal", "Kalpana", "Kamini", "Kanan",
        "Kinnari", "Komal", "Kumud", "Kusum", "Lata", "Laxmi",
        "Leela", "Madhu", "Maitri", "Mansi", "Mira", "Mittal",
        "Monika", "Mrudula", "Naina", "Nalini", "Nandini", "Nayana",
        "Neela", "Nidhi", "Nila", "Nisha", "Nita", "Pallavi",
        "Panna", "Parul", "Pooja", "Poorvi", "Prabha", "Pragna",
        "Priti", "Purvi", "Pushpa", "Rachana", "Radha", "Rani",
        "Reena", "Rekha", "Renuka", "Rupa", "Rupal", "Sarla",
        "Savita", "Sejal", "Shilpa", "Shobhna", "Shreya", "Smriti",
        "Sneha", "Sonal", "Sonali", "Sudha", "Surbhi", "Swati",
        "Tara", "Tarla", "Usha", "Varsha", "Vatsala", "Vibha",
        "Vidya", "Vinita", "Yamini", "Zalak", "Archi", "Bhakti",
    ],

    # ---- NORTH INDIA (Punjabi / Sikh) ----
    "punjabi_male": [
        "Gurpreet", "Harpreet", "Manpreet", "Jaspreet", "Kuldeep",
        "Hardeep", "Sukhdeep", "Randeep", "Amarjeet", "Paramjeet",
        "Jagjeet", "Ranjeet", "Navjot", "Harbhajan", "Yuvraj",
        "Amritpal", "Bhagwant", "Charanjit", "Daljit", "Gurbir",
        "Harjinder", "Iqbal", "Jaswinder", "Karanbir", "Lakhvir",
        "Maninder", "Narinder", "Prabhjot", "Ravinder", "Satinder",
        "Tejinder", "Gagandeep", "Balwinder", "Sukhwinder", "Kulwinder",
        # --- expanded ---
        "Ajitpal", "Amanpal", "Angad", "Arjun", "Avtar",
        "Baba", "Bachittar", "Bahadur", "Balbir", "Baldev",
        "Balkar", "Balraj", "Bhupinder", "Bikram", "Birpal",
        "Chamkaur", "Charan", "Chattar", "Dara", "Darshan",
        "Davinder", "Dharminder", "Dilbag", "Dilawar", "Fauja",
        "Gajinder", "Giani", "Gobind", "Gurdas", "Gurdial",
        "Gurfateh", "Gurlal", "Gurmeet", "Gurmukh", "Gurnam",
        "Gurnoor", "Gurpal", "Gursharan", "Gurtej", "Gyan",
        "Hardyal", "Hargobind", "Harkishan", "Harmeet", "Harnaam",
        "Harpal", "Harwinder", "Inderpal", "Inderjit", "Jagatjit",
        "Jagmeet", "Jagtar", "Jaimal", "Jasbir", "Jaskaran",
        "Jaswant", "Jatinder", "Joginder", "Jugraj", "Karnail",
        "Karamjit", "Kashmeera", "Kehar", "Kharak", "Kirpal",
        "Kuljit", "Kulmeet", "Labh", "Lakhbir", "Livtar",
        "Maghar", "Mahinder", "Mehtab", "Mohan", "Mohkam",
        "Mukhtiar", "Nachattar", "Nihal", "Nirmal", "Onkar",
        "Pargat", "Partap", "Piara", "Pritam", "Rajbir",
        "Rajpal", "Ranjit", "Resham", "Roop", "Sarabjit",
        "Sarbjit", "Sardara", "Sarup", "Sewa", "Shahbaz",
        "Sham", "Sohan", "Sucha", "Sujan", "Sukha",
        "Sukhbir", "Sukhpal", "Surjit", "Tarlok", "Tarsem",
        "Ujagar", "Uttam", "Virsa", "Zorawar", "Amarpal",
        "Balpreet", "Chanpreet", "Devinder", "Ekam", "Fateh",
        "Gurveer", "Harsimran", "Ikjot", "Jaskirat", "Kanwar",
        "Livjot", "Meharban", "Nirvair", "Prabhdeep", "Rajveer",
    ],
    "punjabi_female": [
        "Gurpreet", "Harpreet", "Manpreet", "Jaspreet", "Simran",
        "Navneet", "Amandeep", "Rupinder", "Harleen", "Jasleen",
        "Kirandeep", "Lovepreet", "Mandeep", "Nimrat", "Prabhjot",
        "Rajveer", "Sukhleen", "Tarneet", "Amrita", "Baljinder",
        "Dilpreet", "Gurleen", "Harsimran", "Jasmeet", "Komalpreet",
        # --- expanded ---
        "Amarjit", "Amolak", "Anamika", "Arshdeep", "Avneet",
        "Balbir", "Baljit", "Beant", "Bhinder", "Charanjit",
        "Daljeet", "Davinder", "Diljit", "Gagan", "Gurbani",
        "Gurdip", "Gurkiran", "Gurjeet", "Gurlal", "Gurminder",
        "Gurnoor", "Gursheen", "Harjeet", "Harkirat", "Harmeet",
        "Harnoor", "Harpal", "Harsharan", "Ikjot", "Inderjit",
        "Jagjit", "Jashanpreet", "Jaskamal", "Jaskiran", "Jasmeen",
        "Jaswant", "Kamaldeep", "Kamaljit", "Karamjeet", "Kiran",
        "Kiranjit", "Kuldeep", "Kulvinder", "Maninder", "Manjit",
        "Manjot", "Mehtab", "Narinder", "Navjeet", "Navnoor",
        "Navpreet", "Nirmal", "Pallavi", "Paramjit", "Pardeep",
        "Parminder", "Pawandeep", "Pooja", "Prabhleen", "Preet",
        "Rajinder", "Randeep", "Ranjit", "Reetinder", "Roshni",
        "Rupali", "Sahibdeep", "Sarabjit", "Sarbjit", "Satnam",
        "Savneet", "Seerat", "Shamsher", "Sharanjit", "Simerjit",
        "Sukhjinder", "Sukhman", "Suneet", "Surinder", "Tajinder",
        "Taranjit", "Tejpal", "Upinder", "Valerie", "Veerpal",
    ],

    # ---- MUSLIM NAMES (Pan-India) ----
    "muslim_male": [
        "Mohammed", "Ahmad", "Ali", "Hassan", "Hussain", "Ibrahim",
        "Irfan", "Imran", "Farhan", "Faisal", "Aamir", "Salman",
        "Shahid", "Rizwan", "Wasim", "Zaheer", "Asif", "Nadeem",
        "Javed", "Tariq", "Khalid", "Rashid", "Sajid", "Tanveer",
        "Arbaaz", "Bilal", "Danish", "Ehsan", "Ghulam", "Hamza",
        "Junaid", "Kashif", "Liaqat", "Moin", "Naushad", "Owais",
        "Parvez", "Rameez", "Sabir", "Usman", "Waqar", "Yusuf",
        "Zeeshan", "Altaf", "Bashir", "Dilshad", "Feroz", "Ghalib",
        "Idris", "Kamran", "Mansoor", "Noman", "Rauf", "Shakeel",
        "Ayan", "Rayaan", "Zayan", "Kabir", "Rehan", "Saad",
        # --- expanded ---
        "Abdul", "Abdullah", "Abubakar", "Adil", "Aejaz", "Afroz",
        "Afzal", "Ahad", "Ajmal", "Akbar", "Akhtar", "Aleem",
        "Amaan", "Aman", "Amjad", "Anees", "Ansar", "Anwar",
        "Aquib", "Arif", "Arshad", "Asad", "Asghar", "Ashfaq",
        "Athar", "Atif", "Ayaan", "Azeem", "Azhar", "Aziz",
        "Babar", "Baqir", "Barkat", "Basit", "Dawood", "Dilwar",
        "Ehtesham", "Ejaz", "Faiz", "Farid", "Farooq", "Fawad",
        "Firoz", "Furqan", "Gaffar", "Ghaus", "Habib", "Hafeez",
        "Haider", "Hameed", "Hammad", "Hanif", "Haris", "Hashim",
        "Humayun", "Ikram", "Ilyas", "Inam", "Iqbal", "Ismail",
        "Izhar", "Jabbar", "Jafar", "Jahangir", "Jalal", "Jamal",
        "Jameel", "Kaleem", "Kamal", "Kareem", "Karim", "Latif",
        "Luqman", "Maalik", "Mahboob", "Majid", "Makhdoom", "Manzoor",
        "Maqbool", "Maqsood", "Masood", "Mazhar", "Mehmood", "Misbah",
        "Moazzam", "Mohsin", "Mudassar", "Mufti", "Mujeeb", "Mukhtar",
        "Mumtaz", "Munir", "Mushtaq", "Mustak", "Mustafa", "Muzaffar",
        "Naeem", "Nafees", "Nahid", "Najam", "Naser", "Nasir",
        "Naveed", "Nazeer", "Nazim", "Nisar", "Noor", "Obaid",
        "Qadir", "Qamar", "Qasim", "Quaiser", "Rafiq", "Rahim",
        "Rahman", "Rais", "Raja", "Rashed", "Riyaz", "Ruhul",
        "Sabeer", "Sadiq", "Safdar", "Sahil", "Saif", "Sajan",
        "Salauddin", "Saleem", "Sameer", "Sarfraz", "Shabbir", "Shafiq",
        "Shahbaz", "Shahnawaz", "Shakil", "Shamim", "Shams", "Shareef",
        "Shoaib", "Siraj", "Sohail", "Sultan", "Tahir", "Talib",
        "Tasleem", "Taufiq", "Ubaid", "Wahab", "Waheed", "Yasin",
        "Yunus", "Zafar", "Zahid", "Zaid", "Zakariya", "Zameer",
        "Zia", "Zubair", "Zulfiqar", "Aariz", "Ahsan", "Burhan",
        "Daniyal", "Fahad", "Gibran", "Haseeb", "Izhaan", "Jibreel",
        "Kais", "Lishan", "Mahir", "Nahyan", "Omar", "Pasha",
        "Raakin", "Saqib", "Taha", "Umair", "Wasi", "Yaqoob",
    ],
    "muslim_female": [
        "Fatima", "Ayesha", "Zainab", "Khadija", "Sana", "Hina",
        "Nazia", "Shabana", "Rukhsar", "Tabassum", "Yasmin", "Zoya",
        "Aliya", "Bushra", "Dilshad", "Farzana", "Gulshan", "Husna",
        "Ishrat", "Jameela", "Kulsum", "Lubna", "Meher", "Nasreen",
        "Parveen", "Razia", "Salma", "Tahira", "Uzma", "Waheeda",
        "Afreen", "Benazir", "Chandni", "Farah", "Heena", "Inaya",
        "Mahira", "Naira", "Sadia", "Amira", "Iqra", "Mariam",
        # --- expanded ---
        "Aamina", "Aasiya", "Afia", "Afroza", "Aleena", "Amna",
        "Anisa", "Anjum", "Arifa", "Asma", "Azra", "Bano",
        "Begum", "Bilkis", "Bibi", "Daulat", "Fakhra", "Fareeda",
        "Fatma", "Feroza", "Firdaus", "Ghazala", "Habiba", "Hajeera",
        "Halima", "Hamida", "Haseena", "Hazra", "Humaira", "Husn",
        "Iram", "Jabeen", "Jahan", "Jahanara", "Jamila", "Jarina",
        "Kauser", "Khanam", "Laila", "Latifa", "Mahjabeen", "Malika",
        "Maryam", "Masarrat", "Masooda", "Mehnaz", "Mumtaz", "Munira",
        "Muneera", "Naaz", "Nabiha", "Naghma", "Naheed", "Najma",
        "Nargis", "Nasima", "Nausheen", "Naveeda", "Nazima", "Nazneen",
        "Neelofar", "Nighat", "Nilofer", "Noor", "Noori", "Nusrat",
        "Pakeeza", "Qamar", "Rafat", "Raheela", "Rahima", "Raisa",
        "Rakhshanda", "Rashida", "Raushan", "Rehana", "Rida", "Roshan",
        "Rubina", "Ruhina", "Rukaya", "Sabiha", "Sadaf", "Saeedah",
        "Safina", "Sahar", "Saima", "Sajida", "Sakeena", "Samina",
        "Sameera", "Shaeen", "Shafia", "Shaheen", "Shahida", "Shaista",
        "Shakira", "Shamim", "Shamsunnahar", "Shanaz", "Shareefa",
        "Shazia", "Shireen", "Siddika", "Sitara", "Suhana", "Sultana",
        "Suraya", "Tahmina", "Tanveer", "Tarannum", "Tasnim", "Tehzeeb",
        "Ulfat", "Wahida", "Yaqoota", "Zahida", "Zahra", "Zakia",
        "Zareena", "Zebunnisa", "Zubaida", "Zulekha", "Aaliya", "Aayat",
        "Aneesa", "Dua", "Erum", "Farida", "Ghufran", "Hamna",
        "Insharah", "Jannat", "Kashaf", "Lamiya", "Madiha", "Nimrah",
    ],

    # ---- CHRISTIAN NAMES (Pan-India, especially Kerala, Goa, NE) ----
    "christian_male": [
        "Joseph", "Thomas", "George", "John", "David", "Samuel",
        "Daniel", "Abraham", "Philip", "James", "Peter", "Paul",
        "Anthony", "Francis", "Michael", "Robert", "William", "Stephen",
        "Mathew", "Alexander", "Benjamin", "Christopher", "Dominic", "Edward",
        "Gabriel", "Kevin", "Lawrence", "Nathan", "Oscar", "Vincent",
        "Alen", "Benny", "Cyril", "Deepu", "Elvis", "Felix",
        # --- expanded ---
        "Aaron", "Adam", "Adrian", "Albert", "Alfred", "Alwyn",
        "Ambrose", "Andrew", "Angelo", "Anson", "Arnold", "Austin",
        "Basil", "Benedict", "Bernard", "Brian", "Calvin", "Cecil",
        "Charles", "Clarence", "Clement", "Clinton", "Colin", "Conrad",
        "Cornelius", "Crispin", "Curtis", "Damien", "Darren", "Dennis",
        "Derek", "Desmond", "Donald", "Douglas", "Duncan", "Edgar",
        "Edwin", "Elias", "Emmanuel", "Eric", "Ernest", "Eugene",
        "Fabian", "Ferdinand", "Fredrick", "Gerald", "Gilbert", "Glen",
        "Godfrey", "Gordon", "Gregory", "Hansel", "Harold", "Harvey",
        "Henry", "Herbert", "Herman", "Hilary", "Howard", "Hugh",
        "Ian", "Ivan", "Jacob", "Jason", "Jeffrey", "Jerome",
        "Joel", "Johnson", "Jonathan", "Jordan", "Joshua", "Julian",
        "Justin", "Keith", "Kenneth", "Lancelot", "Leonard", "Leslie",
        "Lewis", "Lionel", "Louis", "Lucas", "Malcolm", "Marcus",
        "Mark", "Martin", "Maurice", "Maxwell", "Melvin", "Miles",
        "Nicholas", "Nigel", "Noel", "Norman", "Oliver", "Oswald",
        "Owen", "Patrick", "Percival", "Percy", "Pierce", "Quentin",
        "Ralph", "Raymond", "Reginald", "Reuben", "Richard", "Robin",
        "Rodney", "Roger", "Roland", "Ronald", "Roy", "Ruben",
        "Sebastian", "Selwyn", "Simon", "Solomon", "Stanley", "Stuart",
        "Sylvester", "Terrence", "Theodore", "Timothy", "Trevor", "Vernon",
        "Victor", "Walter", "Warren", "Wesley", "Wilfred", "Xavier",
        "Zachary", "Agustin", "Bosco", "Cruz", "Desouza", "Elroy",
        "Fernanado", "Gaspar", "Ignatius", "Joaquim", "Lorenzo", "Marcelo",
        "Nazareth", "Pascual", "Rafael", "Salvatore", "Tiago", "Valerian",
    ],
    "christian_female": [
        "Mary", "Rose", "Grace", "Elizabeth", "Sarah", "Rachel",
        "Ruth", "Rebecca", "Esther", "Hannah", "Lydia", "Martha",
        "Anna", "Catherine", "Diana", "Emily", "Florence", "Gloria",
        "Helen", "Irene", "Jessica", "Karen", "Linda", "Margaret",
        "Nancy", "Patricia", "Sheela", "Teresa", "Veronica", "Alice",
        "Anita", "Brinda", "Celine", "Diya", "Elsa", "Fiona",
        # --- expanded ---
        "Agnes", "Alexandra", "Amelia", "Andrea", "Angela", "Angelina",
        "Beatrice", "Bernadette", "Bertha", "Bridget", "Camilla", "Carol",
        "Caroline", "Charlotte", "Christina", "Clara", "Claudia", "Colleen",
        "Constance", "Cynthia", "Daphne", "Deborah", "Delilah", "Dolores",
        "Dorothy", "Edith", "Eleanor", "Elena", "Erica", "Eugenia",
        "Eva", "Evelyn", "Faith", "Felicia", "Frances", "Gabriella",
        "Geraldine", "Gertrude", "Giselle", "Gladys", "Hazel", "Heidi",
        "Hilda", "Hope", "Isabelle", "Ivy", "Jacqueline", "Janet",
        "Janice", "Jennifer", "Joan", "Josephine", "Joyce", "Judith",
        "Julia", "Juliana", "Justine", "Kathleen", "Lara", "Laura",
        "Lavinia", "Lena", "Lillian", "Lois", "Lorraine", "Louisa",
        "Lucia", "Lucy", "Mabel", "Magdalene", "Maria", "Marion",
        "Marlene", "Matilda", "Maureen", "Melanie", "Mercy", "Michelle",
        "Mildred", "Miranda", "Monica", "Natalie", "Nicole", "Nora",
        "Olivia", "Pamela", "Pauline", "Penelope", "Phyllis", "Priscilla",
        "Regina", "Rhoda", "Rita", "Rosalind", "Rosemary", "Ruby",
        "Sabrina", "Sandra", "Selina", "Sharon", "Sheila", "Shirley",
        "Silvia", "Sophia", "Stella", "Susanna", "Sylvia", "Tabitha",
        "Thelma", "Theresa", "Tracy", "Ursula", "Valerie", "Vanessa",
        "Victoria", "Violet", "Virginia", "Wendy", "Winifred", "Yvonne",
        "Zita", "Agatha", "Bernice", "Cecilia", "Dolly", "Elvira",
        "Francisca", "Gracie", "Immaculata", "Josette", "Letitia", "Merlyn",
    ],

    # ---- NORTHEAST INDIA ----
    "northeast_male": [
        "Bhaichung", "Lalrinnunga", "Mirabai", "Nongdren", "Thangkhiew",
        "Lalremsiama", "Jeje", "Sunil", "Rongmei", "Haokip",
        "Dingko", "Bijendra", "Sushil", "Shyam", "Tarundeep",
        "Lovlina", "Hima", "Konsam", "Thangjam", "Laishram",
        "Khumanthem", "Ngangom", "Oinam", "Sapam", "Thokchom",
        # --- expanded ---
        "Alemba", "Amos", "Angam", "Arbin", "Baichung", "Balajied",
        "Bamon", "Banshanglang", "Bareh", "Bhabok", "Bhim", "Bijoy",
        "Birbal", "Chinglen", "Chinglensana", "Churachand", "Dayal",
        "Debabrata", "Devjit", "Dhyan", "Donboklang", "Dondor",
        "Ebormi", "Ghatotkacha", "Gouramangi", "Ibomcha", "Ibson",
        "Imoba", "Irom", "Jarnail", "Jewel", "Joydeep", "Kangabam",
        "Keithelakpam", "Kh", "Khaidem", "Khumukcham", "Kimkhen",
        "Kipgen", "Koijam", "Kongbrailatpam", "Konthoujam", "Lenthoi",
        "Letminthang", "Longam", "Luikham", "Maisnam", "Malemngamba",
        "Manglem", "Naocha", "Naobam", "Naoroibam", "Nepram",
        "Ningthoujam", "Nongmeikapam", "Okram", "Paonam", "Phijam",
        "Pukhrambam", "Rajkumar", "Ringai", "Saikhom", "Sarangthem",
        "Seiminlen", "Singson", "Soibam", "Soraisham", "Thangal",
        "Thingnam", "Thoudam", "Thounaojam", "Tomba", "Tongbram",
        "Waikhom", "Wahengbam", "Warepam", "Yaikhom", "Yanglem",
        "Amu", "Anungla", "Atokha", "Bendang", "Chuba",
        "Diehun", "Easterine", "Hekani", "Imchen", "Jenito",
        "Kekhriesato", "Meren", "Neiphiu", "Pele", "Rongsen",
        "Temjen", "Vikho", "Wapang", "Yhunshalo", "Zhaleo",
        "Aiban", "Banshan", "Donbok", "Ferdinand", "Hamlet",
        "Kenny", "Lariot", "Micky", "Pynhun", "Ridlang",
        "Shanbor", "Teiboklang", "Wanbok", "Bah", "Lyngdoh",
    ],
    "northeast_female": [
        "Hima", "Mary", "Lovlina", "Mirabai", "Saikhom", "Elina",
        "Bembem", "Devi", "Jamuna", "Kshetrimayum", "Laishram",
        "Moirangthem", "Naorem", "Ongbi", "Pukhrambam", "Sagolsem",
        # --- expanded ---
        "Abenla", "Akham", "Angom", "Aribam", "Athokpam", "Ayekpam",
        "Bidyarani", "Bilashini", "Bimola", "Bobita", "Chanambam",
        "Chanu", "Chongtham", "Debala", "Heisnam", "Hijam",
        "Huirem", "Imoinu", "Irengbam", "Khomdram", "Khundrakpam",
        "Konsam", "Kumari", "Langpoklakpam", "Laibi", "Leima",
        "Leishangthem", "Maibam", "Maibi", "Manglem", "Memcha",
        "Metchinla", "Moirangthem", "Naobi", "Nganbi", "Nongmaithem",
        "Oinam", "Pangambam", "Phuritsabam", "Rajkumari", "Sapam",
        "Sharungbam", "Soibam", "Tampha", "Tamphasana", "Telem",
        "Thiyam", "Thoudam", "Tombi", "Tongbram", "Waikhom",
        "Asenla", "Bendangla", "Chubala", "Imsunaro", "Keviseno",
        "Lanusenla", "Merenla", "Neidonuo", "Prunuo", "Sentila",
        "Temsunaro", "Vimenuo", "Watila", "Aphi", "Baia",
        "Daphi", "Ibadahun", "Lakyrsiew", "Merrily", "Onerisa",
        "Phibahun", "Rindalin", "Shaimon", "Walanbha", "Eyingtung",
    ],

    # ---- RAJASTHANI ----
    "rajasthani_male": [
        "Bhanwar", "Chotu", "Daulat", "Gajendra", "Hanuman", "Jagdish",
        "Kanhaiya", "Laxman", "Mahavir", "Nathu", "Pappu", "Raghunath",
        "Shambhu", "Tulsi", "Vishnu", "Bheru", "Gordhan", "Inder",
        "Jagram", "Kailash", "Manohar", "Omprakash", "Pukhraj", "Ratan",
        # --- expanded ---
        "Amarsingh", "Arjunsingh", "Babulal", "Badarilal", "Bahadursingh",
        "Balbir", "Baldev", "Balkishan", "Bansilal", "Bhanwarlal",
        "Bhawani", "Bheekham", "Bihari", "Brijmohan", "Chaturbhuj",
        "Chhaganlal", "Chhitar", "Chotulal", "Dalchand", "Daulatram",
        "Devilal", "Dhanraj", "Dharamchand", "Durgashankar", "Fateh",
        "Ganpat", "Ghewar", "Girraj", "Gokulchand", "Goverdhan",
        "Gulabchand", "Hardayal", "Harisingh", "Hazari", "Heeralal",
        "Hemraj", "Hukumchand", "Jairam", "Jawaharlal", "Jethmal",
        "Jodhsingh", "Jugalkishore", "Kaluram", "Kanaram", "Kesarimal",
        "Khetaram", "Kishorilal", "Lalchand", "Liladhar", "Lunkaran",
        "Madanlal", "Maganlal", "Mannalal", "Mathuralal", "Meghraj",
        "Mohanlal", "Motilal", "Murarilal", "Nagraj", "Nandkishore",
        "Nanuram", "Narayansingh", "Narottam", "Pannalal", "Parasram",
        "Pokhar", "Pratap", "Puranchand", "Pyarelal", "Radheshyam",
        "Rajmal", "Ramniwas", "Ramswaroop", "Ranjeet", "Ratanlal",
        "Rughnath", "Sajjan", "Sampatlal", "Sanwarmal", "Sardarmal",
        "Satyanarayan", "Shankarlal", "Shivdayal", "Shrawan", "Sitaram",
        "Sohanraj", "Sujanmal", "Sundarlal", "Swaroopsingh", "Tejsingh",
        "Tolaram", "Udairam", "Ummedsingh", "Vikram", "Virendra",
    ],
    "rajasthani_female": [
        "Bhanwari", "Champa", "Durga", "Gomti", "Hansa", "Jamuna",
        "Kaushalya", "Lakshmi", "Mira", "Panna", "Radha", "Santosh",
        "Tara", "Vidya", "Basanti", "Chhagan", "Devu", "Gyarsi",
        # --- expanded ---
        "Amari", "Anandi", "Anokhi", "Badami", "Baijanti", "Bali",
        "Banno", "Bhagwati", "Bhuri", "Bibbi", "Chandan", "Chandrakala",
        "Choti", "Dhanki", "Dhapubai", "Gangabai", "Gaurja", "Geejgar",
        "Gumani", "Hansi", "Heera", "Hirabai", "Indra", "Janki",
        "Jasoda", "Jhuma", "Kali", "Kalki", "Kamla", "Kesar",
        "Khatu", "Kishni", "Lali", "Leelabai", "Lilabai", "Mangibai",
        "Meethi", "Mishri", "Moti", "Nani", "Narbada", "Parvati",
        "Phooli", "Premi", "Rajbala", "Rajkumari", "Rami", "Ratan",
        "Rodhi", "Roopkanwar", "Rukmani", "Sagar", "Sajni", "Samri",
        "Saraswati", "Saroj", "Savitri", "Sita", "Sugna", "Sumitra",
        "Sunder", "Sundri", "Surja", "Teeja", "Tulsa", "Ugami",
        "Vandana", "Vimla", "Badri", "Chand", "Dhapu", "Ganga",
        "Hari", "Jagdamba", "Kailashi", "Ladu", "Mangi", "Nanhi",
        "Pappi", "Ramkali", "Shakuntala", "Teja", "Urja", "Devki",
    ],

    # ---- JAIN ----
    "jain_male": [
        "Mahavir", "Rishabh", "Adinath", "Parshvanath", "Nemichand",
        "Chandraprabha", "Jinesh", "Kundan", "Padmanabh", "Shreyansh",
        "Vardhamana", "Hemchand", "Shantilal", "Ratanlal", "Babulal",
        # --- expanded ---
        "Abhinandan", "Ajitnath", "Amarchand", "Amritlal", "Anantvirya",
        "Atmanand", "Bhadrabahu", "Bhagchand", "Bharmal", "Bhikchand",
        "Champatlal", "Chandanbala", "Chandulal", "Chhaganlal", "Chhotalal",
        "Dalpatram", "Devichand", "Dharamdas", "Dhirajlal", "Dulichand",
        "Ganeshmal", "Gautamlal", "Gheesalal", "Gulabchand", "Harichand",
        "Hastimal", "Hirachand", "Hiralal", "Inderlal", "Jaichand",
        "Jainulabdin", "Jitmal", "Jugalkishor", "Kaluram", "Kantilal",
        "Kasturchand", "Keshrichand", "Khemlal", "Lalchand", "Laxmichand",
        "Liladhar", "Madanlal", "Mangilal", "Mannalal", "Mathuralal",
        "Meghchand", "Mohanlal", "Motilal", "Mulchand", "Namonarayan",
        "Nandalal", "Nathmal", "Navratan", "Neminath", "Niranjan",
        "Omprakash", "Pannalal", "Parasmal", "Prahalad", "Premchand",
        "Prithviraj", "Punamchand", "Raichand", "Rajmal", "Ramchand",
        "Ratnasagar", "Rupesh", "Sagarmal", "Sanghvi", "Sanjivlal",
        "Sanmati", "Santlal", "Shaligram", "Shanmal", "Shripat",
        "Sohanraj", "Subhash", "Suganchand", "Sundarlal", "Swarnchand",
    ],
    "jain_female": [
        "Jinaya", "Padmini", "Trishala", "Chandana", "Marudevi",
        "Nandini", "Riddhima", "Sthanakvasi", "Yashodhara", "Prabhavati",
        # --- expanded ---
        "Ahimsa", "Ambalika", "Anokhi", "Archana", "Bhagwati",
        "Champalal", "Champaben", "Damyanti", "Dayawanti", "Draupadi",
        "Geetadevi", "Gomti", "Heera", "Hemlataben", "Induben",
        "Jamnaben", "Kanchan", "Kantaben", "Karunaben", "Kastur",
        "Kiran", "Kokilaben", "Kunti", "Labhuben", "Lata",
        "Lilaben", "Madhuriben", "Manbai", "Mangalben", "Meenaben",
        "Motiben", "Muktaben", "Nirupama", "Panna", "Premlataben",
        "Pushpa", "Rajulben", "Ratanbai", "Rekha", "Rupalben",
        "Santokben", "Sarojben", "Savitaben", "Shakuntala", "Shantaben",
        "Sharda", "Shobhna", "Sumitradevi", "Susilaben", "Taraben",
        "Triveniben", "Ujamben", "Urmila", "Vasantben", "Vimalben",
    ],

    # ---- PARSI / ZOROASTRIAN ----
    "parsi_male": [
        "Rustom", "Cyrus", "Darius", "Hormuz", "Jamshed", "Kaikhosru",
        "Nadir", "Pheroze", "Shapoor", "Zubin", "Adi", "Boman",
        "Farokh", "Homi", "Jehangir", "Nusli", "Ratan", "Sam",
        # --- expanded ---
        "Ardeshir", "Arnavaz", "Aspandiar", "Bahram", "Behram",
        "Behroze", "Bejan", "Berjis", "Burjor", "Darab",
        "Dinshaw", "Edulji", "Erach", "Erachshaw", "Farrokh",
        "Firdausi", "Firoz", "Framroze", "Fredoon", "Godrej",
        "Gustad", "Hoshang", "Hormazd", "Jal", "Jamsheed",
        "Jamshedji", "Kaikobad", "Kaizad", "Keki", "Kersi",
        "Khurshed", "Kobad", "Maneck", "Mehelli", "Mehernosh",
        "Merwan", "Minocher", "Minoo", "Nadir", "Nariman",
        "Noshir", "Nowroji", "Pallonji", "Parvez", "Pesi",
        "Pirojsha", "Ratansha", "Rohinton", "Ronnie", "Rumi",
        "Rustam", "Sarosh", "Savak", "Sohrab", "Sorabji",
        "Spenta", "Tehmton", "Tehemten", "Viraf", "Vispi",
        "Xerxes", "Yazdi", "Zarir", "Zarin", "Zervaan",
        "Adil", "Bakhtiar", "Cooverji", "Dara", "Edulji",
        "Faredoon", "Govad", "Hirji", "Irani", "Jivaji",
        "Khojeste", "Lovji", "Manchersha", "Nowshir", "Pervez",
    ],
    "parsi_female": [
        "Anahita", "Bakhtavar", "Dilnavaz", "Freny", "Gool", "Homai",
        "Jeroo", "Kashmira", "Meher", "Nergish", "Persis", "Roshan",
        "Shirin", "Tehmi", "Villoo", "Yasmin", "Zenobia", "Arnavaz",
        # --- expanded ---
        "Aloo", "Armaity", "Arnawaz", "Banoo", "Bapsy",
        "Benaifer", "Coomi", "Daulat", "Dhun", "Dinaz",
        "Dina", "Dossi", "Ervad", "Farida", "Farzana",
        "Firuza", "Freni", "Gulbanoo", "Gulnar", "Gulshanbanoo",
        "Havovi", "Hilla", "Hutokshi", "Jaloo", "Jamshed",
        "Jerbanoo", "Kainaaz", "Kamal", "Katayun", "Kety",
        "Khorshed", "Khursheed", "Manijeh", "Marazban", "Mehroo",
        "Mithoo", "Mithra", "Naju", "Navaz", "Noshir",
        "Perin", "Perveen", "Piloo", "Putli", "Ratanbanoo",
        "Roda", "Roxana", "Rukhshana", "Rustomji", "Sarvar",
        "Shahbanoo", "Shehernaz", "Sheroo", "Shiraz", "Spenta",
        "Tehmina", "Tenaz", "Tina", "Vehesta", "Veera",
        "Zarine", "Zenab", "Zenia", "Zubeida", "Zarin",
        "Aspy", "Behroz", "Cyra", "Delna", "Ervad",
        "Firooza", "Goher", "Hursh", "Izar", "Jerbai",
    ],

    # ---- ASSAMESE ----
    "assamese_male": [
        "Bhupen", "Jyoti", "Lakshminath", "Hem", "Nabakanta",
        "Phani", "Rongmon", "Srimanta", "Bishnu", "Dwipen",
        "Gaurav", "Hrishikesh", "Indrajit", "Kamal", "Mridul",
        # --- expanded ---
        "Ajit", "Amar", "Ananta", "Arup", "Ashim", "Atul",
        "Babul", "Baharul", "Bhaskar", "Bhrigu", "Bijoy", "Bikram",
        "Bimal", "Binod", "Biplab", "Bipul", "Biren", "Bolin",
        "Budhindra", "Chandranath", "Cheniram", "Deben", "Devajit",
        "Dharma", "Dhiren", "Diganta", "Dilip", "Dinesh",
        "Dipak", "Durlov", "Ghanakanta", "Girindra", "Gobin",
        "Gunajit", "Guru", "Hari", "Hemanga", "Hitesh",
        "Jadav", "Jagadish", "Jiten", "Jogen", "Jogesh",
        "Jugal", "Kamakhya", "Kanak", "Karabi", "Keshab",
        "Khagen", "Kushal", "Laban", "Lakhi", "Madhab",
        "Mahananda", "Mahendra", "Manik", "Manuj", "Mohan",
        "Mukul", "Munin", "Nabajyoti", "Nagen", "Nalini",
        "Naren", "Niranjan", "Nirod", "Nripen", "Padma",
        "Paran", "Paresh", "Prafulla", "Pranab", "Prasanta",
        "Pratul", "Putul", "Raju", "Ramen", "Rana",
        "Ranjit", "Rupam", "Sailen", "Sankar", "Sarat",
        "Sashi", "Simanta", "Tarun", "Tilak", "Tirthankar",
        "Utpal", "Zubin", "Abhinab", "Bikash", "Chandan",
    ],
    "assamese_female": [
        "Arundhati", "Bonti", "Chayanika", "Dimple", "Erina",
        "Gargi", "Himadri", "Jutika", "Kankana", "Lipika",
        "Mallika", "Nibedita", "Papori", "Rina", "Subasana",
        # --- expanded ---
        "Aarti", "Anamika", "Ankita", "Anupama", "Archana",
        "Banalata", "Bandita", "Barasha", "Barnali", "Bidyut",
        "Bijayashree", "Bobita", "Bornali", "Chameli", "Chandana",
        "Chitralekha", "Deepanjali", "Dipali", "Dulumoni", "Gitali",
        "Gitanjali", "Gungun", "Hemjyoti", "Hiyamoni", "Indrani",
        "Jahnabi", "Jayashree", "Jinti", "Jonaki", "Jumi",
        "Junali", "Kaberi", "Kakali", "Kalindi", "Kamala",
        "Karabi", "Kasturi", "Kiran", "Konwari", "Kumud",
        "Lakhimi", "Majoni", "Malabika", "Mamoni", "Manashi",
        "Manalisha", "Manju", "Meenakshi", "Mitali", "Moloya",
        "Monalisha", "Mridula", "Nabanita", "Nalini", "Namita",
        "Nandita", "Niharika", "Nilakshi", "Nirmali", "Niru",
        "Pallabi", "Parinita", "Parveen", "Polly", "Pori",
        "Prarthana", "Pratima", "Pratishtha", "Prerana", "Priyanka",
        "Puja", "Rashmi", "Renu", "Rima", "Rituparna",
        "Roshmi", "Rupali", "Saheli", "Sangita", "Saoni",
        "Seuti", "Silpi", "Smita", "Soma", "Subarna",
        "Sumitra", "Sunanda", "Suparna", "Tapasi", "Tilottama",
    ],
}


# ============================================================================
# SURNAMES / LAST NAMES -- organized by region
# ============================================================================

SURNAMES = {
    "hindi_general": [
        "Sharma", "Verma", "Gupta", "Agarwal", "Jain", "Mittal",
        "Bansal", "Goel", "Singhal", "Rastogi", "Saxena", "Srivastava",
        "Mishra", "Pandey", "Tiwari", "Dubey", "Shukla", "Dwivedi",
        "Tripathi", "Chaturvedi", "Upadhyay", "Pathak", "Bajpai", "Dixit",
        "Kumar", "Singh", "Yadav", "Chauhan", "Rajput", "Thakur",
        "Rawat", "Negi", "Bisht", "Joshi", "Bhatt", "Pant",
        "Chandra", "Prakash", "Prasad", "Lal", "Chand", "Dayal",
        # --- expanded ---
        "Nigam", "Khanna", "Bhatnagar", "Kaushik", "Taneja", "Anand",
        "Kapoor", "Wadhwa", "Mehra", "Khatri", "Ahuja", "Malhotra",
        "Bhatia", "Arora", "Chawla", "Sethi", "Dhawan", "Sachdeva",
        "Grover", "Gulati", "Walia", "Oberoi", "Mahajan", "Khurana",
        "Vohra", "Soni", "Sood", "Chopra", "Luthra", "Bajaj",
        "Tyagi", "Nagar", "Gaur", "Goswami", "Vashishtha", "Saraswat",
        "Agnihotri", "Kulshreshtha", "Parashar", "Misra", "Ojha", "Dikshit",
        "Awasthi", "Vajpayee", "Mukherji", "Gangwar", "Paliwal", "Kanaujia",
        "Bist", "Dobhal", "Chamoli", "Gairola", "Nautiyal", "Semwal",
        "Juyal", "Kimothi", "Raturi", "Pokhriyal", "Gahtori", "Bhandari",
        "Kunwar", "Dandriyal", "Rawaliya", "Kala", "Khatik", "Dhangar",
        "Chandel", "Sengar", "Bais", "Gahlot", "Panwar", "Solanki",
        "Bundela", "Bargoti", "Rathod", "Deshwal", "Dalal", "Sangwan",
        "Sheoran", "Panghal", "Sehrawat", "Tanwar", "Ahlawat", "Dabas",
        "Jakhar", "Kadian", "Mann", "Punia", "Hooda", "Lamba",
        "Saini", "Gujjar", "Beniwal", "Jangra", "Saran", "Poonia",
        "Chaudhary", "Kamboj", "Dhaka", "Godara", "Kaswan", "Saharan",
        "Nehra", "Berwal", "Balyan", "Dedha", "Gora", "Kataria",
        "Malik", "Rangi", "Rana", "Bisla", "Gill", "Bhardwaj",
        "Rathore", "Shekhawat", "Tandon", "Kapur", "Puri", "Narang",
    ],
    "hindi_obc_sc": [
        "Yadav", "Kushwaha", "Maurya", "Patel", "Lodhi", "Rajbhar",
        "Nishad", "Bind", "Kashyap", "Saini", "Meena", "Gurjar",
        "Jatav", "Valmiki", "Paswan", "Ram", "Manjhi", "Musahar",
        "Dhobi", "Khatik", "Chamar", "Ahirwar", "Sonkar", "Pal",
        # --- expanded ---
        "Kumhar", "Lohar", "Teli", "Tamoli", "Barai", "Barhai",
        "Darzi", "Dhimar", "Gaddi", "Gadaria", "Gadariya", "Kalal",
        "Kalwar", "Kewat", "Koeri", "Kurmi", "Luniya", "Mallah",
        "Nai", "Noniya", "Pasi", "Prajapati", "Rajak", "Sonar",
        "Vishwakarma", "Badhai", "Bhujwa", "Chauhan", "Dhanuk", "Dusadh",
        "Gaur", "Halwai", "Jaiswal", "Kahar", "Kannaujia", "Kanu",
        "Kesarwani", "Khatri", "Kori", "Mali", "Modanwal", "Nonia",
        "Pathan", "Rawani", "Sahu", "Shah", "Sunar", "Swarnakar",
        "Tanti", "Thathera", "Ahir", "Dhangar", "Gupta", "Kalsi",
        "Kumawat", "Mahato", "Nonia", "Rajwade", "Saket", "Tekam",
        "Uike", "Vankar", "Waghmare", "Bairwa", "Chamdia", "Dhurve",
        "Gond", "Jhariya", "Kol", "Munda", "Oraon", "Pradhan",
        "Rohidas", "Santal", "Thakur", "Vanskar", "Warkade", "Gahane",
    ],

    "punjabi": [
        "Singh", "Kaur", "Gill", "Sidhu", "Dhillon", "Sandhu",
        "Grewal", "Bajwa", "Brar", "Cheema", "Deol", "Hayer",
        "Johal", "Khera", "Mann", "Randhawa", "Sahota", "Thind",
        "Ahluwalia", "Bhatia", "Chopra", "Kapoor", "Malhotra", "Oberoi",
        "Sethi", "Tandon", "Wadhwa", "Arora", "Chawla", "Khurana",
        "Soni", "Talwar", "Vohra", "Walia", "Anand", "Bedi",
        # --- expanded ---
        "Aulakh", "Athwal", "Basra", "Bath", "Bhatti", "Bhullar",
        "Boparai", "Buttar", "Chahal", "Chana", "Chattha", "Dayal",
        "Dhaliwal", "Dhanoa", "Ghuman", "Ghumman", "Goraya", "Hundal",
        "Jawanda", "Jhand", "Kahlon", "Kalra", "Kalsi", "Kang",
        "Khangura", "Khatkar", "Khosa", "Kooner", "Lakhvir", "Lalli",
        "Lehal", "Liddar", "Maan", "Mahal", "Mangat", "Mavi",
        "Minhas", "Multani", "Nagra", "Nahal", "Nijjar", "Panag",
        "Panesar", "Parhar", "Pelia", "Phull", "Rai", "Riar",
        "Romana", "Sahotra", "Samra", "Sandhawalia", "Sangha", "Sarkaria",
        "Sekhon", "Shergill", "Sohi", "Sran", "Toor", "Uppal",
        "Virdi", "Virk", "Ahuja", "Bawa", "Bhalla", "Chadha",
        "Dhingra", "Duggal", "Ghai", "Gujral", "Hansra", "Handa",
        "Jaggi", "Jaitly", "Juneja", "Kakkar", "Kanwar", "Kohli",
        "Lamba", "Luthra", "Madan", "Mehra", "Narang", "Narula",
        "Pasricha", "Puri", "Sabharwal", "Sahni", "Sehgal", "Sikand",
        "Sodhi", "Suri", "Takhar", "Thapar", "Trehan", "Tuteja",
    ],

    "bengali": [
        "Banerjee", "Chatterjee", "Mukherjee", "Bhattacharya", "Ganguly",
        "Ghosh", "Bose", "Sen", "Roy", "Das", "Dutta", "Mitra",
        "Chakraborty", "Sarkar", "Biswas", "Mondal", "Haldar", "Kundu",
        "Saha", "Paul", "Nandi", "Goswami", "Majumdar", "Bagchi",
        "Chowdhury", "De", "Kar", "Pal", "Adhikari", "Barua",
        # --- expanded ---
        "Acharya", "Baidya", "Bandyopadhyay", "Basak", "Basu", "Bhowmick",
        "Chakrabarti", "Choudhury", "Dam", "Dasgupta", "Deb", "Debnath",
        "Ghoshal", "Guha", "Gupta", "Halder", "Jana", "Karmakar",
        "Lahiri", "Maiti", "Mallick", "Mandal", "Mazumder", "Mitra",
        "Mukherjee", "Nag", "Naskar", "Panda", "Poddar", "Pramanik",
        "Purkayastha", "Raychaudhuri", "Roychoudhury", "Saha", "Samanta",
        "Sanyal", "Seal", "Sengupta", "Shil", "Sil", "Singha",
        "Som", "Talukdar", "Thakur", "Tiwari", "Bhar", "Boral",
        "Burman", "Chattopadhyay", "Dandapat", "Dolui", "Gain", "Ghorai",
        "Hazra", "Kamilya", "Konar", "Maity", "Midya", "Murmu",
        "Nandy", "Paramanik", "Pati", "Raha", "Rajak", "Saren",
        "Sasmal", "Sau", "Tudu", "Barik", "Dhali", "Kayal",
        "Khatun", "Kotal", "Mahapatra", "Mahata", "Naik", "Panja",
        "Sasmal", "Senapati", "Hansda", "Hembram", "Kisku", "Mardi",
        "Soren", "Baski", "Besra", "Sing", "Murmu", "Tudu",
    ],

    "marathi": [
        "Patil", "Deshmukh", "Jadhav", "Pawar", "Shinde", "More",
        "Kulkarni", "Deshpande", "Joshi", "Gokhale", "Bhagwat", "Apte",
        "Sathe", "Phadke", "Kelkar", "Ranade", "Karve", "Tilak",
        "Chavan", "Gaikwad", "Kamble", "Waghmare", "Bhosale", "Thorat",
        "Kale", "Mane", "Salunkhe", "Shirke", "Rane", "Thakare",
        # --- expanded ---
        "Bapat", "Barve", "Bhide", "Chitnis", "Dabholkar", "Damle",
        "Dandekar", "Date", "Gadgil", "Ghaisas", "Gore", "Gupte",
        "Jog", "Joglekar", "Kale", "Kanitkar", "Karmarkar", "Keer",
        "Ketkar", "Khandekar", "Kolhatkar", "Kunte", "Lele", "Limaye",
        "Madgaonkar", "Mahajan", "Mandlik", "Marathe", "Modak", "Moghe",
        "Muley", "Naik", "Natu", "Oak", "Pandit", "Paranjape",
        "Parkhe", "Phanse", "Potdar", "Puranik", "Rajwade", "Ranade",
        "Sapkal", "Sarpotdar", "Savarkar", "Sowani", "Tambe", "Tamhane",
        "Tendulkar", "Thite", "Vaidya", "Velankar", "Vichare", "Wadkar",
        "Wagh", "Atre", "Bhave", "Diwan", "Gharpure", "Inamdar",
        "Kadam", "Khaire", "Mohite", "Nimbalkar", "Phule", "Sardar",
        "Sutar", "Ubale", "Valunjkar", "Yewale", "Ahire", "Bagul",
        "Bhandare", "Bhoir", "Dalvi", "Dhavale", "Gavhane", "Ghule",
        "Ingale", "Jedhe", "Kharat", "Landge", "Magar", "Nalawade",
        "Ovhal", "Pagare", "Raut", "Shelke", "Tekale", "Ugale",
    ],

    "gujarati": [
        "Patel", "Shah", "Mehta", "Desai", "Modi", "Trivedi",
        "Joshi", "Pandya", "Bhatt", "Dave", "Shukla", "Vyas",
        "Parikh", "Amin", "Naik", "Thakkar", "Kothari", "Doshi",
        "Gandhi", "Shroff", "Parekh", "Soni", "Mistry", "Dalal",
        "Gajjar", "Makwana", "Solanki", "Rathod", "Vaghela", "Jadeja",
        # --- expanded ---
        "Acharya", "Adhvaryu", "Barot", "Bharwad", "Bhavsar", "Brahmbhatt",
        "Chauhan", "Chokshi", "Darji", "Dholakia", "Divetia", "Doshi",
        "Gohel", "Gosai", "Goswami", "Harijan", "Iyer", "Jhaveri",
        "Kanani", "Kapadia", "Khatri", "Kotak", "Lakhani", "Luhar",
        "Majmudar", "Maniar", "Munshi", "Nayak", "Panchal", "Pandey",
        "Parekh", "Prajapati", "Raval", "Rawal", "Ruparel", "Sanghvi",
        "Sarvaiya", "Sheth", "Shukl", "Suthar", "Tanna", "Thaker",
        "Trivedi", "Upadhyay", "Vasavada", "Zaveri", "Amin", "Bhanushali",
        "Chudasama", "Dabhi", "Engineer", "Fadia", "Gadhavi", "Hathi",
        "Inamdar", "Jhala", "Kansara", "Lodhia", "Madhani", "Nagori",
        "Oza", "Parmar", "Raiyani", "Saraiya", "Thakor", "Vekariya",
        "Waghela", "Yadav", "Zala", "Baxi", "Contractor", "Diwan",
        "Firodia", "Ghanchi", "Hingorani", "Jethva", "Keshwala", "Limbani",
        "Mevawala", "Nagar", "Padhiar", "Rabari", "Sagathia", "Tailor",
    ],

    "tamil": [
        "Iyer", "Iyengar", "Mudaliar", "Pillai", "Nadar", "Thevar",
        "Gounder", "Chettiar", "Reddiar", "Nair", "Pandiyan", "Raja",
        "Subramanian", "Krishnamurthy", "Venkataraman", "Sundaram",
        "Natarajan", "Ramaswamy", "Shanmugam", "Palaniswamy",
        "Selvaraj", "Murugesan", "Arumugam", "Balasubramanian",
        # --- expanded ---
        "Alagappan", "Anbazhagan", "Arunachalam", "Ashok", "Balamurugan",
        "Chelladurai", "Chinnadurai", "Deivanayagam", "Durai", "Duraiswamy",
        "Ganapathy", "Gnanasekaran", "Gurusamy", "Ilangovan", "Jeyachandran",
        "Kailasam", "Kalyanasundaram", "Kandasamy", "Kathiresan", "Kumaresan",
        "Lakshmanan", "Loganathan", "Mahalingam", "Malaichamy", "Manickam",
        "Marimuthu", "Meganathan", "Muthukumar", "Muthusamy", "Nagarajan",
        "Nallasivam", "Palanisamy", "Perumal", "Ponnuswamy", "Rajagopal",
        "Rajakumar", "Rajendran", "Ramachandran", "Ramanathan", "Ranganathan",
        "Rathinavelu", "Ravichandran", "Sambandam", "Saminathan", "Saravanan",
        "Sathyamoorthy", "Selvaraju", "Senthilkumar", "Shanmugasundaram",
        "Sivasubramanian", "Somasundaram", "Soundararajan", "Srinivasan",
        "Subramaniam", "Sundaresan", "Thangavel", "Thirunavukkarasu",
        "Udayar", "Vaithianathan", "Velayutham", "Venugopal", "Viswanathan",
        "Veerappan", "Velusamy", "Yoganathan", "Chidambaram", "Devarajan",
        "Elangovan", "Ganesan", "Hariharan", "Ilanchezhian", "Jayaraman",
        "Kothandaraman", "Lakshmi", "Manoharan", "Namasivayam", "Palani",
    ],

    "telugu": [
        "Reddy", "Rao", "Naidu", "Choudhary", "Varma", "Raju",
        "Goud", "Kamma", "Setty", "Mudaliar", "Kapu", "Nayak",
        "Murthy", "Prasad", "Krishna", "Chary", "Rani", "Devi",
        "Swamy", "Babu", "Sastry", "Sharma", "Achary", "Pantulu",
        # --- expanded ---
        "Agarwal", "Appa", "Apparao", "Bhaskar", "Bhimasena", "Brahma",
        "Challagundla", "Chandra", "Chintala", "Dasari", "Devarakonda",
        "Dommeti", "Gadde", "Gajula", "Galla", "Gandikota", "Ganta",
        "Gudivada", "Gunapati", "Gunturi", "Kakani", "Kakinada",
        "Kambhampati", "Kancharla", "Kandula", "Karri", "Katragadda",
        "Kodali", "Kola", "Kondapalli", "Korrapati", "Kotha", "Kunapuli",
        "Maddala", "Majeti", "Malladi", "Mandava", "Mangipudi", "Medarametla",
        "Meka", "Modukuri", "Mogili", "Mynampati", "Nallamothu", "Nandyala",
        "Nemani", "Nutakki", "Oruganti", "Pagidipalli", "Palaparthi",
        "Parimi", "Penmetsa", "Pidugu", "Pinninti", "Poduri", "Pokuri",
        "Pothuri", "Raghavendra", "Rayapudi", "Rekulapalli", "Samudrala",
        "Sattiraju", "Siripurapu", "Surampudi", "Tangirala", "Tenali",
        "Thotapalli", "Uppalapati", "Vadlamudi", "Valluri", "Vemula",
        "Vempati", "Vemulapalli", "Yerraguntla", "Yerra", "Yeturu",
    ],

    "kannada": [
        "Gowda", "Hegde", "Shetty", "Bhat", "Nayak", "Rao",
        "Acharya", "Prabhu", "Kamath", "Pai", "Kulkarni", "Patil",
        "Desai", "Joshi", "Shastri", "Murthy", "Swamy", "Reddy",
        # --- expanded ---
        "Adiga", "Ballal", "Bangera", "Bhandary", "Bhat", "Devadiga",
        "Gowda", "Gudigar", "Iyengar", "Iyer", "Kini", "Kodancha",
        "Kudva", "Maiya", "Mallya", "Moolya", "Moily", "Naik",
        "Nayar", "Pail", "Poojary", "Prabhu", "Salian", "Shenoy",
        "Shanbhag", "Suvarna", "Tantry", "Udupa", "Upadhya", "Bagi",
        "Basappa", "Byahatti", "Channabasappa", "Deshmukh", "Gadad",
        "Havaldar", "Hiremath", "Hublikar", "Indi", "Jamkhandi",
        "Kaddi", "Khanapur", "Koppal", "Lakkundi", "Managuli",
        "Nargund", "Panchamasali", "Raichur", "Saundatti", "Shirahatti",
        "Torgal", "Uppin", "Vibhuti", "Wali", "Yaranal",
        "Adke", "Baliga", "Byndoor", "Ganiga", "Heroor",
        "Karkera", "Kotian", "Maniyani", "Rao", "Sequeira",
    ],

    "malayalam": [
        "Nair", "Menon", "Pillai", "Kurup", "Panikkar", "Varma",
        "Namboothiri", "Iyer", "Potti", "Thampi", "Kartha", "Kaimal",
        "Thomas", "Joseph", "Mathew", "George", "Abraham", "Kurian",
        "Philip", "Chacko", "Varghese", "Pothen", "Oommen", "Zachariah",
        # --- expanded ---
        "Achuthan", "Balakrishnan", "Chandran", "Damodaran", "Divakaran",
        "Easwaran", "Gangadharan", "Gopalan", "Govindan", "Harikumar",
        "Jayachandran", "Karunakaran", "Krishnakumar", "Krishnan", "Kumaran",
        "Lakshmanan", "Madhavan", "Mohanan", "Muraleedharan", "Narayanan",
        "Padmanabhan", "Parameswaran", "Prabhakaran", "Radhakrishnan",
        "Raghu", "Raghunathan", "Rajagopalan", "Rajasekharan", "Raman",
        "Ravindranath", "Sasidharan", "Sekharan", "Shanmughan", "Sivadasan",
        "Sreedharan", "Subramaniam", "Surendran", "Thankappan", "Unnikrishnan",
        "Velayudhan", "Vijayan", "Cherian", "D'Souza", "Fernandez",
        "Idicula", "Jacob", "John", "Jose", "Koshy",
        "Lukose", "Mammen", "Mathai", "Paul", "Poulose",
        "Samuel", "Simon", "Solomon", "Tharakan", "Titus",
        "Varkey", "Antony", "Babu", "Daniel", "David",
        "Elias", "Gabriel", "Isaac", "James", "Kurien",
        "Mathews", "Ninan", "Ouseph", "Peter", "Roy",
        "Sebastian", "Stephen", "Sunny", "Xavier", "Yohannan",
    ],

    "muslim": [
        "Khan", "Siddiqui", "Ansari", "Sheikh", "Pathan", "Mirza",
        "Qureshi", "Hashmi", "Rizvi", "Faruqi", "Jafri", "Naqvi",
        "Baig", "Malik", "Syed", "Hussain", "Ahmed", "Ali",
        "Beg", "Chishti", "Dehlvi", "Gilani", "Haider", "Jilani",
        "Kazmi", "Lucknowi", "Momin", "Nomani", "Osmani", "Qazi",
        # --- expanded ---
        "Abbasi", "Abidi", "Akhtar", "Alvi", "Amrohi", "Ashrafi",
        "Azmi", "Badayuni", "Bakshi", "Banuri", "Bilgrami", "Bukhari",
        "Changezi", "Darbari", "Deobandi", "Faizabadi", "Firangi", "Ghaznavi",
        "Ghori", "Hamdani", "Hasani", "Hyderabadi", "Idrisi", "Illahabadi",
        "Isfahani", "Jaunpuri", "Kakori", "Kandhlawi", "Karimi", "Kashmiri",
        "Kidwai", "Korangi", "Lakhnavi", "Madani", "Mahfooz", "Maududi",
        "Mehmood", "Mewati", "Mohani", "Mubarakpuri", "Nadwi", "Nakhuda",
        "Narnauli", "Naumani", "Niazi", "Palampuri", "Peshawari", "Pirzada",
        "Rahmani", "Rehmani", "Saharanpuri", "Salafi", "Sambhali", "Sarwari",
        "Shamsi", "Sherwani", "Siddiqi", "Surti", "Tablighi", "Thanvi",
        "Tonki", "Usmani", "Warsi", "Zaidi", "Zamani", "Zia",
        "Alig", "Bihari", "Chapra", "Danapur", "Gazipur", "Islahi",
        "Kirmani", "Meeruti", "Pilibhiti", "Rampur", "Sultanpur", "Tanda",
    ],

    "northeast": [
        "Chhetri", "Tamang", "Rai", "Gurung", "Lepcha", "Bhutia",
        "Subba", "Pradhan", "Sharma", "Thapa", "Lama", "Sherpa",
        "Meitei", "Haokip", "Kipgen", "Vaiphei", "Ralte", "Khiangte",
        "Dkhar", "Lyngdoh", "Marbaniang", "Syiem", "Rymbai", "Kharkongor",
        # --- expanded ---
        "Angami", "Ao", "Baruah", "Bodo", "Boro", "Chakma",
        "Chang", "Chiru", "Debbarma", "Dimasa", "Garo", "Hajong",
        "Hmar", "Jamatia", "Kabui", "Karbi", "Khasi", "Koch",
        "Konyak", "Kuki", "Liangmai", "Lushei", "Mao", "Maring",
        "Mizo", "Monsang", "Naga", "Paite", "Phom", "Poumai",
        "Rabha", "Rengma", "Reang", "Sangtam", "Sema", "Singpho",
        "Tangkhul", "Thadou", "Tiwa", "Wancho", "Zeliang", "Zeme",
        "Aimol", "Anal", "Biate", "Chothe", "Gangte", "Kom",
        "Lamkang", "Maram", "Moyon", "Purum", "Simte", "Tarao",
        "Thangal", "Vaiphei", "Zou", "Basumatary", "Brahma", "Daimary",
        "Khakhlary", "Mahilary", "Musahary", "Narzary", "Ramchiary",
        "Swargiary", "Wary", "Bhumij", "Deori", "Khampti", "Mishing",
    ],

    "rajasthani": [
        "Shekhawat", "Rathore", "Chauhan", "Gehlot", "Meena", "Gurjar",
        "Jat", "Rajpurohit", "Vyas", "Maheshwari", "Agarwal", "Khandelwal",
        "Surana", "Somani", "Daga", "Chouhan", "Bhati", "Sisodia",
        # --- expanded ---
        "Bairwa", "Balai", "Berwa", "Bhambhi", "Chamar", "Chipa",
        "Daroga", "Devasi", "Dewasi", "Dhakad", "Dhanka", "Godara",
        "Gujjar", "Jakhar", "Jangid", "Jat", "Kachhwaha", "Kalbi",
        "Kaswan", "Kumawat", "Kumhar", "Lakhara", "Lohar", "Luhar",
        "Mali", "Mev", "Mina", "Nai", "Ojha", "Panchariya",
        "Parihar", "Poonia", "Purohit", "Rajawat", "Ranawat", "Rebari",
        "Regar", "Saini", "Salvi", "Sankhla", "Saran", "Sharma",
        "Shekhawat", "Sindhi", "Siyag", "Solanki", "Suthar", "Tanwar",
        "Teli", "Tomar", "Udawat", "Verma", "Vishnoi", "Yadav",
        "Banjara", "Bheel", "Damami", "Garasiya", "Kanjar", "Nat",
        "Sansi", "Bagri", "Dhobi", "Gadri", "Mochi", "Nath",
        "Raika", "Rawat", "Sindhi", "Chippa", "Chhipa", "Darji",
    ],

    "odia": [
        "Mohanty", "Mishra", "Panda", "Dash", "Sahoo", "Nayak",
        "Behera", "Jena", "Patra", "Swain", "Pradhan", "Rout",
        "Sahu", "Sethi", "Biswal", "Patnaik", "Tripathy", "Senapati",
        # --- expanded ---
        "Acharya", "Barik", "Bastia", "Bhoi", "Bisoi", "Bose",
        "Chand", "Dalai", "Deo", "Dharua", "Gadnayak", "Giri",
        "Gouda", "Guru", "Hota", "Jagdev", "Kanhar", "Kar",
        "Khatua", "Khuntia", "Kisan", "Lenka", "Maharana", "Majhi",
        "Malla", "Meher", "Mohapatra", "Muduli", "Nanda", "Naik",
        "Palai", "Parida", "Pati", "Raita", "Rath", "Ray",
        "Routray", "Sahu", "Samantaray", "Samantray", "Satapathy",
        "Sarangi", "Singh", "Subudhi", "Suna", "Sundari", "Tandi",
        "Tripathy", "Baral", "Bhanja", "Dalabehera", "Dehuri", "Ghadai",
        "Hansda", "Jali", "Karji", "Marndi", "Nahak", "Paik",
        "Sanhura", "Tandi", "Bag", "Bej", "Chinara", "Digal",
        "Gagarai", "Hembram", "Jhodia", "Khara", "Lakra", "Majhi",
    ],

    "parsi": [
        "Tata", "Wadia", "Godrej", "Mistry", "Irani", "Patel",
        "Bhabha", "Desai", "Engineer", "Commissariat", "Anklesaria",
        "Batliwala", "Daruwala", "Gazdar", "Kapadia", "Panthaki",
        # --- expanded ---
        "Aga", "Amaria", "Antia", "Bamji", "Baria", "Barucha",
        "Batliboi", "Bharucha", "Billimoria", "Birdy", "Boyce", "Captain",
        "Chinoy", "Choksy", "Cama", "Contractor", "Cooper", "Dadabhoy",
        "Dalal", "Dastoor", "Davar", "Deboo", "Doctor", "Dotivala",
        "Driver", "Dubash", "Dumasia", "Edalji", "Elavia", "Erach",
        "Fatakia", "Framjee", "Gandhy", "Ghadiali", "Ginwala", "Gowadia",
        "Havewala", "Homji", "Hormuzd", "Jal", "Javeri", "Jeejeebhoy",
        "Jehangir", "Jussawala", "Katrak", "Kerawala", "Khambata", "Kotwal",
        "Lashkari", "Madan", "Maneckji", "Marshall", "Mazda", "Mehta",
        "Minocherhomji", "Mirza", "Modi", "Mogrelia", "Motiwala", "Nagarwala",
        "Nariman", "Nazir", "Nowrojee", "Parakh", "Pavri", "Petit",
        "Pochkhanawala", "Readymoney", "Sarkari", "Sethna", "Sidhwa",
        "Suntook", "Taraporewala", "Todywala", "Umrigar", "Vakil",
    ],
}


# ============================================================================
# COMMON TITLES / PREFIXES (sometimes appear in UPI)
# ============================================================================

TITLES = ["Mr", "Mrs", "Smt", "Shri", "Dr", "Sri", "Kumari"]

# ============================================================================
# NAME GENERATION LOGIC
# ============================================================================

def get_compatible_surnames(region_key: str) -> list[str]:
    """Map a first-name region to compatible surname groups."""
    mapping = {
        "hindi": ["hindi_general", "hindi_obc_sc", "rajasthani"],
        "tamil": ["tamil"],
        "telugu": ["telugu"],
        "kannada": ["kannada"],
        "malayalam": ["malayalam"],
        "bengali": ["bengali"],
        "odia": ["odia"],
        "marathi": ["marathi"],
        "gujarati": ["gujarati"],
        "punjabi": ["punjabi"],
        "muslim": ["muslim"],
        "christian": ["malayalam", "tamil", "bengali", "hindi_general"],
        "northeast": ["northeast"],
        "rajasthani": ["rajasthani", "hindi_general"],
        "jain": ["gujarati", "hindi_general", "rajasthani"],
        "parsi": ["parsi"],
        "assamese": ["bengali", "northeast"],
    }

    # Extract region from key (e.g., "hindi_male" -> "hindi")
    region = region_key.rsplit("_", 1)[0]

    surname_keys = mapping.get(region, ["hindi_general"])
    result = []
    for key in surname_keys:
        result.extend(SURNAMES.get(key, []))
    return result


def generate_name_variants(first_name: str, surname: str, gender: str) -> list[dict]:
    """Generate multiple UPI-realistic format variants of a single name."""
    variants = []

    # 1. Standard: "Firstname Lastname"
    variants.append({
        "name": f"{first_name} {surname}",
        "format": "firstname_lastname",
    })

    # 2. ALL CAPS: "FIRSTNAME LASTNAME"
    variants.append({
        "name": f"{first_name} {surname}".upper(),
        "format": "upper",
    })

    # 3. all lower: "firstname lastname"
    variants.append({
        "name": f"{first_name} {surname}".lower(),
        "format": "lower",
    })

    # 4. Initial + Lastname: "R Sharma"
    variants.append({
        "name": f"{first_name[0]} {surname}",
        "format": "initial_lastname",
    })

    # 5. Firstname + Initial: "Ramesh S"
    variants.append({
        "name": f"{first_name} {surname[0]}",
        "format": "firstname_initial",
    })

    # 6. First name only: "Ramesh"
    variants.append({
        "name": first_name,
        "format": "firstname_only",
    })

    # 7. Lastname Firstname (South Indian style): "Sharma Ramesh"
    variants.append({
        "name": f"{surname} {first_name}",
        "format": "lastname_firstname",
    })

    return variants


def generate_three_part_names(first_name: str, middle_name: str, surname: str) -> list[dict]:
    """Generate three-part name variants (common in Hindi belt)."""
    variants = []

    # "Ramesh Kumar Sharma"
    variants.append({
        "name": f"{first_name} {middle_name} {surname}",
        "format": "three_part",
    })

    # "RAMESH KUMAR SHARMA"
    variants.append({
        "name": f"{first_name} {middle_name} {surname}".upper(),
        "format": "three_part_upper",
    })

    # "R K Sharma"
    variants.append({
        "name": f"{first_name[0]} {middle_name[0]} {surname}",
        "format": "initials_lastname",
    })

    return variants


# Common middle names (Hindi belt)
MIDDLE_NAMES_MALE = [
    "Kumar", "Prakash", "Chandra", "Nath", "Kishore", "Mohan",
    "Shankar", "Bahadur", "Prasad", "Lal", "Chand", "Raj",
    "Deo", "Kant", "Deep", "Pal", "Ratan", "Dayal",
    # --- expanded ---
    "Narayan", "Gopal", "Bihari", "Ballabh", "Lochan", "Mani",
    "Murari", "Bhushan", "Sagar", "Vardhan", "Vallabh", "Mohan",
    "Bhan", "Dhar", "Dev", "Deen", "Ram", "Sharan",
    "Swaroop", "Roop", "Priya", "Tej", "Veer", "Jeet",
    "Hari", "Madhav", "Keshav", "Govind", "Shyam", "Giri",
    "Pati", "Das", "Din", "Nandan", "Anand", "Sukh",
    "Bir", "Jit", "Inder", "Partap", "Sahai", "Sewak",
    "Dutt", "Dhir", "Man", "Parkash", "Nand", "Bhagat",
    "Sunder", "Kishan", "Bansi", "Trilok", "Puran", "Jagat",
    "Lachhman", "Brij", "Daya", "Har", "Madan", "Padam",
    "Sher", "Udai", "Fateh", "Nihal", "Bachan", "Kewal",
    "Piara", "Chhote", "Lambu", "Guddu", "Bhola", "Tikam",
]
MIDDLE_NAMES_FEMALE = [
    "Kumari", "Devi", "Rani", "Lata", "Mala", "Prabha",
    "Jyoti", "Rekha", "Shree", "Bala", "Sundari", "Lakshmi",
    # --- expanded ---
    "Bai", "Wati", "Mati", "Vati", "Kaur", "Begum",
    "Bibi", "Khatoon", "Nissa", "Fatima", "Jan", "Kala",
    "Dulari", "Pyari", "Sundri", "Premi", "Gauri", "Daya",
    "Vidya", "Kanti", "Smriti", "Pushpa", "Chandra", "Mohini",
    "Padma", "Malti", "Kamla", "Shanti", "Shobha", "Sushila",
    "Nirmala", "Saraswati", "Savitri", "Parvati", "Durga", "Ganga",
    "Yamuna", "Maya", "Radha", "Sita", "Geeta", "Leela",
    "Hema", "Sarla", "Sudha", "Vimala", "Kiran", "Rupa",
    "Indu", "Tara", "Sneha", "Priti", "Asha", "Usha",
]


def generate_dataset(
    names_per_region: int = 200,
    variants_per_name: int = 3,
    three_part_ratio: float = 0.2,
    seed: int = 42,
) -> list[dict]:
    """
    Generate the full Indian names dataset.

    Args:
        names_per_region: How many unique first+last combos per region/gender group
        variants_per_name: How many format variants to sample per name
        three_part_ratio: Fraction of names that get three-part variants too
        seed: Random seed for reproducibility

    Returns:
        List of dicts with keys: name, label, gender, region, format
    """
    random.seed(seed)
    dataset = []
    seen_names = set()  # deduplicate

    for region_key, first_names in FIRST_NAMES.items():
        # Determine gender from key
        gender = "female" if region_key.endswith("_female") else "male"

        # Get compatible surnames
        compatible_surnames = get_compatible_surnames(region_key)
        if not compatible_surnames:
            continue

        for _ in range(names_per_region):
            first = random.choice(first_names)
            last = random.choice(compatible_surnames)

            # Generate format variants
            all_variants = generate_name_variants(first, last, gender)

            # Optionally add three-part names
            if random.random() < three_part_ratio:
                middle_list = MIDDLE_NAMES_MALE if gender == "male" else MIDDLE_NAMES_FEMALE
                middle = random.choice(middle_list)
                all_variants.extend(
                    generate_three_part_names(first, middle, last)
                )

            # Sample a subset of variants
            sampled = random.sample(
                all_variants,
                min(variants_per_name, len(all_variants))
            )

            for variant in sampled:
                name_lower = variant["name"].lower().strip()
                if name_lower not in seen_names:
                    seen_names.add(name_lower)
                    dataset.append({
                        "name": variant["name"],
                        "label": "PERSON",
                        "gender": gender,
                        "region": region_key.rsplit("_", 1)[0],
                        "format": variant["format"],
                    })

    # Shuffle
    random.shuffle(dataset)

    return dataset


def save_dataset(dataset: list[dict], output_path: str):
    """Save dataset to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ["name", "label", "gender", "region", "format"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    print(f"[OK] Saved {len(dataset)} names to {output_path}")


def print_stats(dataset: list[dict]):
    """Print dataset statistics."""
    from collections import Counter

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"\nTotal names: {len(dataset)}")

    # By region
    region_counts = Counter(row["region"] for row in dataset)
    print(f"\nBy Region ({len(region_counts)} regions):")
    for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
        print(f"  {region:20s} -> {count:5d}")

    # By gender
    gender_counts = Counter(row["gender"] for row in dataset)
    print(f"\nBy Gender:")
    for gender, count in sorted(gender_counts.items()):
        print(f"  {gender:20s} -> {count:5d}")

    # By format
    format_counts = Counter(row["format"] for row in dataset)
    print(f"\nBy Format ({len(format_counts)} formats):")
    for fmt, count in sorted(format_counts.items(), key=lambda x: -x[1]):
        print(f"  {fmt:25s} -> {count:5d}")

    # Sample
    print(f"\nSample (first 20):")
    for row in dataset[:20]:
        print(f"  {row['name']:30s} | {row['region']:12s} | {row['gender']:6s} | {row['format']}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "indian_person_names.csv")

    dataset = generate_dataset(
        names_per_region=500,    # 500 combos per region/gender group (increased)
        variants_per_name=3,     # 3 format variants each
        three_part_ratio=0.2,    # 20% get three-part names too
        seed=42,
    )

    save_dataset(dataset, output_path)
    print_stats(dataset)
