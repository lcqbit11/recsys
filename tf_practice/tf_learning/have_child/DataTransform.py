#!/usr/bin/env python
# -*- coding: utf-8 -*-

class DataTransform(object):
    def __init__(self, libsvm_data, csv_data, records_data):
        self.libsvm_data = libsvm_data
        self.csv_data = csv_data

