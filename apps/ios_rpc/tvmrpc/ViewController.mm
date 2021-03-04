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
#include <string>
#include "rpc_server.h"

@implementation ViewController

- (void)open {
  NSLog(@"Connecting to the proxy server..");
  
  key_ = [self.proxyKey.text UTF8String];
  port_ = [self.proxyPort.text intValue];
  url_ = [self.proxyURL.text UTF8String];
  
  std::ostringstream ss;
  ss << "('" << url_ << "', " << port_<< ")";
  
  std::string host = "0.0.0.0";
  int port = 9000;
  int port_end = 9099;
  std::string tracker_addr = ss.str();
  std::string custom_addr = "";
  
  // Start the rpc server
  rpc_ = std::make_shared<tvm::runtime::RPCServer>(host, port, port_end, tracker_addr, key_, custom_addr);
  rpc_->Start();
  
  self.infoText.text = @"";
  self.statusLabel.text = @"Connecting...";
}

- (void)close {
  NSLog(@"Closing the streams.");
  if (rpc_ != nullptr)
    rpc_->Stop();
  self.statusLabel.text = @"Disconnected";
}

- (IBAction)connect:(id)sender {
  [self open];
//  [[self view] endEditing:YES];
}

- (IBAction)disconnect:(id)sender {
  [self close];
}

- (void)onShutdownReceived {
  [self close];
}

@end
