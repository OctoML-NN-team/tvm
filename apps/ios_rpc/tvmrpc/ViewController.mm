/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file ViewController.mm
 */

#import "ViewController.h"

#include "rpc_server.h"
#include <string>

@implementation ViewController

- (void)onShutdownReceived {
  [self close];
}

- (void)viewDidLoad {
  self.proxyURL.delegate = self;
  self.proxyPort.delegate = self;
  self.proxyKey.delegate = self;
  
  [self readPreferences];
  [self open];
}

- (BOOL)textFieldShouldReturn:(UITextField *)textField {
  [self updatePreferences];
  [[self view] endEditing:YES];
  return FALSE;
}

- (void)readPreferences {
  NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
  
  self.proxyURL.text = [defaults stringForKey:@"tmvrpc_url"];
  self.proxyPort.text = [defaults stringForKey:@"tmvrpc_port"];
  self.proxyKey.text = [defaults stringForKey:@"tmvrpc_key"];
}

- (void)updatePreferences {
  NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    
  [defaults setObject:self.proxyURL.text forKey:@"tmvrpc_url"];
  [defaults setObject:self.proxyPort.text forKey:@"tmvrpc_port"];
  [defaults setObject:self.proxyKey.text forKey:@"tmvrpc_key"];
}

- (void)open {
  NSLog(@"Connecting to the tracker server..");
  
  std::string key_ = "i12";//[self.proxyKey.text UTF8String];
  std::string url_ = "192.168.0.201";//[self.proxyURL.text UTF8String];
  int port_ = 9190;//[self.proxyPort.text intValue];

  std::ostringstream ss;
  ss << "('" << url_ << "', " << port_<< ")";
  
  std::string host = "0.0.0.0";
  int port = 9000;
  int port_end = 9099;
  std::string tracker_addr = ss.str();
  std::string custom_addr = "";
  std::string work_dir = [NSTemporaryDirectory() UTF8String];
  
  // Start the rpc server
  rpc_ = std::make_shared<tvm::runtime::RPCServer>(host, port, port_end, tracker_addr, key_,
//  rpc_ = std::make_shared<tvm::runtime::RPCServer>(host, port, port_end, "192.168.0.92:9190", "i12",
                                                   custom_addr, work_dir);
  rpc_->setCompletionCallbacks(
    [self] () {
      dispatch_sync(dispatch_get_main_queue(), ^{
        self.statusLabel.text = @"Connected";
      });
    }, [self] () {
      dispatch_sync(dispatch_get_main_queue(), ^{
        self.statusLabel.text = @"Disconnected";
      });
    });
  rpc_->Start();

  self.infoText.text = @"";
  self.statusLabel.text = @"Connecting...";
}


- (void)close {
  NSLog(@"Closing the streams.");
  rpc_->Stop();
}

- (IBAction)connect:(id)sender {
  [self open];
  [[self view] endEditing:YES];
}

- (IBAction)disconnect:(id)sender {
  [self close];
}

@end
