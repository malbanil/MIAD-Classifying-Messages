# MIAD-Classifying-Messages

Classifying Incoming Customer Messages for an E-Commerce Site using Supervised Learning

Abstract. 
Throughout the world, the provision of online goods and services
has increased significantly over the last few years. We consider the
case of Tango Discos, a small company in Colombia that sells entertainment
products through an e-commerce website and receives customer
messages through various channels, including a webform, email, Facebook
and Twitter. This dataset comprises 29,970 messages collected from
2019 to 2021. Each message can be categorized as being either being a
sale, request or complaint. In this work we evaluate different supervised
classification models to automate the task of classifying the messages,
viz. decision trees, Naive Bayes, linear Support Vector Machines and
logistic regression. As the data set is unbalanced, the different models
are evaluated in combination with various data balancing approaches to
obtain the best performance. In order to maximize revenue, the management
is interested in prioritizing messages that may result in potential
sales. As such, the best model for deployment is one that minimizes false
positives in the sales category, so that these are processed in a timely
fashion. As such, the best performing model is found to be the Linear
Support Vector Machine using the Random Over Sampler balancing technique.
This model is deployed in the cloud and exposed using a RESTful
interface.

Keywords: E-commerce · Message classification · Supervised learning
· Support Vector Machine · Balancing techniques

Authors: 
- Misael Andrey Albanil Sanchez[0000−0003−2746−1299]
- Ixent Galpin[0000−0001−7020−6328]

Facultad de Ciencias Naturales e Ingenieria
Universidad de Bogota – Jorge Tadeo Lozano
Bogota, Colombia
{misaela.albanils,ixent}@utadeo.edu.co
