extern crate opencl;

use opencl::Platform;

fn main() {
    for platform in Platform::all().iter() {
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
            println!("   Global Memory Size: {} MB", device.global_mem_size() / (1024 * 1024));
            println!("   Local Memory Size: {} MB", device.local_mem_size() / (1024 * 1024));
            println!("   Max Alloc Size: {} MB", device.max_mem_alloc_size() / (1024 * 1024));
        }
    }
}
