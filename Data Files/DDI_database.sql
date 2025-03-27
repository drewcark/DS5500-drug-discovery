create database if not exists DDI;
use DDI;
create table if not exists interactions (
	int_code int not null,
    int_reduced_code int,
    int_desc mediumtext not null,
    primary key (int_code)
);

create table if not exists inputs (
    inp_num int not null,
    molecule1 varchar(2800),
    molecule2 varchar(2800),
    result int,
    primary key (inp_num),
    foreign key (result) references interactions(int_code)
);
