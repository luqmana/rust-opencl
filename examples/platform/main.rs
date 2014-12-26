extern crate opencl;

use opencl::hl;

fn main() {
    for platform in hl::get_platforms().iter() {
        println!("Platform: {}", platform.name());
        println!("Vendor:   {}", platform.vendor());
        println!("Profile:  {}", platform.profile());
        println!("Available extensions: {}", platform.extensions());
        println!("Available devices:");
        for device in platform.get_devices().iter() {
            println!("   {}", device.name());
        }
    }
}
