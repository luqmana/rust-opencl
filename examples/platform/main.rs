extern crate opencl;

use opencl::hl;

fn main() {
    for platform in hl::get_platforms().iter() {
        println!("Platform: {}", platform.name());
        println!("Platform Version: {}", platform.version());
        println!("Vendor:   {}", platform.vendor());
        println!("Profile:  {}", platform.profile());
        println!("Available extensions: {}", platform.extensions());
        println!("Available devices:");
        for device in platform.get_devices().iter() {
            println!("   Name: {}", device.name());
            println!("   Type: {}", device.device_type());
            println!("   Profile: {}", device.profile());
            println!("   Compute Units: {}", device.compute_units());
            println!("   Global Mem Cache Size: {} Bytes", device.global_mem_cache_size());
            println!("   Global Mem Size: {} Bytes", device.global_mem_size());
            println!("   Local Mem Size: {} Bytes", device.local_mem_size());
            println!("   Max Constant Buffer Size: {} Bytes", device.max_constant_buffer_size());
            println!("   Max Mem Alloc Size: {} Bytes", device.max_mem_alloc_size());
        }
    }
}
