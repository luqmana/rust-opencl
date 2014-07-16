

extern crate debug;

pub fn main(){
	let x = &mut [1i,2,3,4,5];
	x.inc();
	println!("{}", x)
}

trait Inc {
	fn inc(&self);
}

impl<'a> Inc for int{
	fn inc(&mut self){
		*self = *self + 1;
	}
}

impl<'a, I: Inc> Inc for &'a mut [I]{
	fn inc(self){
		println!("{:?}", self)

		let slice = self.as_mut_slice();

		for ref mut part in slice.mut_iter(){
			part.inc();
		}
	}
}